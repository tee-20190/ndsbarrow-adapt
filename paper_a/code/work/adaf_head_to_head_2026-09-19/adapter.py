"""Auditable CLI adapter around unedited official ADAF/AiTLAS model definitions.

Avoids installing the unrelated GUI/training/database dependencies of AiTLAS.
See PLAN.md for the explicitly adapted centroid ranking and evaluation protocol.
"""
from pathlib import Path
import ast,collections,copy,hashlib,json,os,time,zipfile
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.features import shapes,geometry_mask
from affine import Affine
from shapely.geometry import shape,mapping
import torch
import rvt.vis
import rvt

# RVT 2.2.1 imports a removed matplotlib colourmap API, unused by SLRM
# normalization. Load the unedited function definitions without that plotting
# import; no normalization or model expression is changed.
_blend_path=Path(rvt.__file__).parent/'blend_func.py'
_blend_tree=ast.parse(_blend_path.read_text())
_blend_tree.body=[n for n in _blend_tree.body if not(isinstance(n,ast.ImportFrom)
    and n.module=='matplotlib.cm' and any(a.name=='get_cmap' for a in n.names))]
_blend_ns={};exec(compile(_blend_tree,str(_blend_path),'exec'),_blend_ns)
normalize_image=_blend_ns['normalize_image']

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
WHEEL=HERE/'vendor/adaf/installation/aitlas-0.0.1-py3-none-any.whl'
WEIGHTS=HERE/'weights/AO_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def official_nodes():
    with zipfile.ZipFile(WHEEL) as z:
        model_source=z.read('aitlas/models/hrnet.py').decode()
        tree=ast.parse(model_source)
        tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.level)
                   and not(isinstance(n,ast.ClassDef) and n.name=='HRNet')]
        ns={};exec(compile(tree,str(WHEEL)+'!hrnet.py','exec'),ns)
        methods={}
        for filename,method in [('aitlas/base/models.py','predict_masks_tiff_probs_binary'),
                                ('aitlas/base/segmentation.py','get_predicted')]:
            source=z.read(filename).decode()
            matches=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==method]
            assert len(matches)==1,(filename,len(matches))
            namespace=dict(torch=torch,np=np,rasterio=rasterio,os=os,copy=copy,collections=collections)
            exec(compile(ast.Module(body=matches,type_ignores=[]),filename,'exec'),namespace)
            methods[method]=namespace[method]
    return ns,methods,hashlib.sha256(model_source.encode()).hexdigest()

def load_model(device='cuda'):
    ns,methods,source_sha=official_nodes()
    model=ns['HRNetModule'](ns['HRNetSegHead'](2),pretrained=False)
    c=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    model.load_state_dict(c['state_dict'],strict=True)
    del c
    model=model.eval().to(device)
    return model,methods,source_sha

def predict(model,a,device='cuda'):
    tensor=torch.tensor(np.repeat(a[None,:,:],3,axis=0),dtype=torch.float32,device=device)[None]
    with torch.inference_mode():
        return torch.sigmoid(model(tensor)['out'])[0,1].cpu().numpy()

def qualify():
    out=HERE/'qualification';out.mkdir(exist_ok=True)
    model,methods,sha=load_model()
    a=np.random.default_rng(20260919).uniform(0,1,(64,64)).astype('float32')
    inp=out/'synthetic_input.tif'
    with rasterio.open(inp,'w',driver='GTiff',height=64,width=64,count=1,dtype='float32',
                       crs='EPSG:27700',transform=Affine(0.5,0,0,0,-0.5,32)) as ds: ds.write(a,1)
    class View:
        def __init__(self):
            self.model=model;self.device='cuda';self.config=type('Config',(),{'threshold':.5})()
        def __call__(self,x):return self.model(x)
    View.get_predicted=methods['get_predicted']
    View.predict_masks_tiff_probs_binary=methods['predict_masks_tiff_probs_binary']
    with torch.inference_mode():
        View().predict_masks_tiff_probs_binary(image_path=str(inp),label='AO',
            data_transforms=lambda x:torch.tensor(x.transpose(2,0,1),dtype=torch.float32),predictions_dir=str(out))
    with rasterio.open(out/'synthetic_input_AO_segmentation_mask_probs.tif') as ds:official=ds.read(1)
    candidate=predict(model,a)
    assert np.array_equal(official,candidate)
    receipt={'status':'PASS','strict_state_dict':True,'official_method_adapter_max_abs_error':float(np.max(abs(official-candidate))),
             'wheel_sha256':digest(WHEEL),'hrnet_source_sha256':sha,'weights_sha256':digest(WEIGHTS),
             'torch':torch.__version__,'cuda':torch.cuda.get_device_name(),'adapter_sha256':digest(__file__)}
    (HERE/'receipts/qualification.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)

def source_view(src):
    if np.allclose(src.res,(.5,.5)):return src
    assert np.allclose(src.res,(1.,1.)),src.res
    return WarpedVRT(src,transform=src.transform*Affine.scale(.5,.5),width=src.width*2,height=src.height*2,
                     resampling=Resampling.bilinear,nodata=float('nan'),dtype='float32')

def probability_raster(path,out,model,phase_m=0.,interior=None):
    started=time.monotonic()
    with rasterio.open(path) as original:
        src=source_view(original)
        height,width=src.height,src.width
        result=np.zeros((height,width),dtype='float32')
        # ADAF's 32 m context is 64 pixels on its 0.5 m grid.
        phase=int(round(phase_m/.5));halo=64;radius=20;tile=1024
        start=phase-tile if phase else 0
        patches=0
        for col in range(start,width,tile):
            for row in range(start,height,tile):
                # WarpedVRT disallows boundless reads: read intersecting extent and pad explicitly.
                x0,y0=col-halo-radius,row-halo-radius
                size=tile+2*halo+2*radius
                data=np.full((size,size),np.nan,dtype='float32')
                xa,ya=max(x0,0),max(y0,0);xb,yb=min(x0+size,width),min(y0+size,height)
                if xa>=xb or ya>=yb:continue
                part=src.read(1,window=Window(xa,ya,xb-xa,yb-ya),masked=True).filled(np.nan).astype('float32')
                data[ya-y0:yb-y0,xa-x0:xb-x0]=part
                if not np.isfinite(data[radius:-radius,radius:-radius]).any():continue
                slrm=rvt.vis.slrm(data,radius_cell=radius)
                norm=normalize_image('slrm',slrm,-.5,.5,'value')[radius:-radius,radius:-radius]
                norm=np.nan_to_num(norm,nan=0.).astype('float32')
                probs=predict(model,norm)
                # Match official 8px trim prior to vectorisation, merging overlap by union.
                probs=probs[8:-8,8:-8]
                x,y=col-halo+8,row-halo+8
                xa,ya=max(x,0),max(y,0);xb,yb=min(x+probs.shape[1],width),min(y+probs.shape[0],height)
                if xa<xb and ya<yb:
                    np.maximum(result[ya:yb,xa:xb],probs[ya-y:yb-y,xa-x:xb-x],out=result[ya:yb,xa:xb])
                patches+=1
        valid=src.read(1,masked=True);valid=np.isfinite(valid.filled(np.nan)) & ~np.ma.getmaskarray(valid)
        result[~valid]=0
        if interior:
            from shapely.geometry import box
            valid &= geometry_mask([mapping(box(*interior))],out_shape=result.shape,transform=src.transform,invert=True)
            result[~valid]=0
        profile=src.profile.copy();profile.update(driver='GTiff',count=1,dtype='float32',nodata=None,
            compress='deflate',predictor=2,tiled=True,blockxsize=256,blockysize=256)
        out.parent.mkdir(parents=True,exist_ok=True)
        with rasterio.open(out,'w',**profile) as dst:dst.write(result,1)
        mask=(result>=.5).astype('uint8')
        objects=[]
        for g,v in shapes(mask,mask=mask.astype(bool),transform=src.transform):
            poly=shape(g);area=poly.area
            roundness=round(4*np.pi*area/(poly.convex_hull.length**2),3) if poly.length else 0
            if area<=30 or roundness<=.4:continue
            win=rasterio.windows.from_bounds(*poly.bounds,src.transform).round_offsets().round_lengths()
            rr,cc=int(win.row_off),int(win.col_off);hh,ww=int(win.height),int(win.width)
            covered=geometry_mask([mapping(poly)],out_shape=(hh,ww),transform=src.window_transform(win),invert=True)
            score=float(result[rr:rr+hh,cc:cc+ww][covered].max())
            centroid=poly.centroid
            objects.append({'id':f'{path.stem}-{len(objects):05d}','easting':centroid.x,'northing':centroid.y,
                'score':score,'area_m2':area,'roundness':roundness,'source':str(path.relative_to(ROOT))})
        receipt={'source':str(path.relative_to(ROOT)),'source_sha256':digest(path),'phase_m':phase_m,
                 'valid_km2':float(valid.sum())*.25/1e6,'patches':patches,'objects':objects,'wall_s':time.monotonic()-started,
                 'output_sha256':digest(out),'adapter_sha256':digest(__file__)}
        if src is not original:src.close()
    return receipt

if __name__=='__main__':
    import sys
    torch.set_num_threads(4)
    if sys.argv[1]=='qualify':qualify()

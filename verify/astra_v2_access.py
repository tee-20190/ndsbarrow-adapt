"""Auditable England access probes and complete NHLE retrieval; no French inputs."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'targets/lidar-archaeology/astra_v2'
DATA = ROOT / 'work/astra_v2/access'
NHLE = 'https://services-eu1.arcgis.com/ZOdPfBS3aqqDYPUQ/ArcGIS/rest/services/National_Heritage_List_for_England_NHLE_v02_VIEW/FeatureServer/6'
WCS = 'https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs'

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def fetch(url, path, params=None):
    start = now()
    r = requests.get(url, params=params, timeout=90)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    rec = dict(url=r.url, retrieved_utc=now(), started_utc=start, status=r.status_code,
               content_type=r.headers.get('content-type'), bytes=len(r.content),
               path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(r.content).hexdigest())
    with (OUT / 'access_log.jsonl').open('a') as f:
        f.write(json.dumps(rec)+'\n')
    r.raise_for_status()
    return r

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    completed=OUT/'labels_receipt.json'
    if completed.exists():
        receipt=json.loads(completed.read_text())
        p=ROOT/receipt['path']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=receipt['sha256']:raise RuntimeError('Existing NHLE snapshot changed')
        print('Existing immutable NHLE snapshot retained:',receipt['feature_count'],receipt['sha256']);return
    for url, name, params in [
        ('https://environment.data.gov.uk/arcgis/rest/services/EA/LIDAR_Composite_1m_DTM_2022/ImageServer/exportImage', 'imageserver_probe.txt',
         dict(f='json',bbox='410000,130000,410420,130420',bboxSR=27700,imageSR=27700,size='420,420',format='tiff',pixelType='F32')),
        (WCS, 'wcs_capabilities.xml',dict(service='WCS',request='GetCapabilities',version='2.0.1')),
        (NHLE, 'nhle_metadata.json',dict(f='json'))]:
        try:
            r=fetch(url,DATA/name,params)
            print(name,r.status_code,len(r.content),flush=True)
        except Exception as e:
            print(name,repr(e),flush=True)
    ids=fetch(NHLE+'/query',DATA/'nhle_ids.json',dict(f='json',where='1=1',returnIdsOnly='true')).json()['objectIds']
    features=[]
    for i in range(0,len(ids),500):
        p=DATA/f'nhle_offset_{i:05d}.json'
        if p.exists():
            page=json.loads(p.read_text())
        else:
            page=fetch(NHLE+'/query',p,dict(f='geojson',where='1=1',orderByFields='OBJECTID',resultOffset=i,resultRecordCount=500,outFields='*',outSR=27700,returnGeometry='true')).json()
        assert 'features' in page, page
        features.extend(page['features'])
        print('NHLE',len(features),'/',len(ids),flush=True)
    assert len(features)==len(ids)
    p=DATA/'nhle_scheduled_monuments.geojson'
    p.write_text(json.dumps(dict(type='FeatureCollection',crs={'type':'name','properties':{'name':'EPSG:27700'}},features=features)))
    (OUT/'labels_receipt.json').write_text(json.dumps(dict(url=NHLE,feature_count=len(features),retrieved_utc=now(),path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()),indent=2))

if __name__ == '__main__':
    main()

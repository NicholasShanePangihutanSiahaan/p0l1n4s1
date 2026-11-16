"""Conservative vertical-trunk verification near a camera candidate."""
import numpy as np


GEOMETRY_DEFAULTS = dict(
    association_radius=0.8, min_points=30, slice_height=0.25,
    min_slice_points=6, ransac_iterations=60, min_radius=0.10, max_radius=0.65,
    max_center_offset=0.6, circle_tolerance=0.035, min_inlier_ratio=0.6,
    min_arc_coverage=0.5, min_slices=4, slice_center_tolerance=0.12,
    min_vertical_span=0.75, surface_tolerance=0.06,
    bottom_percentile=3.0, top_percentile=97.0, min_height=0.8)


def validate_geometry(config):
    for key,value in config.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be positive and finite')
    for key in ('min_points','min_slice_points','ransac_iterations','min_slices'):
        if int(config[key]) != config[key]: raise ValueError(f'{key} must be an integer')
    if config['min_slice_points'] < 3 or config['min_slices'] < 2:
        raise ValueError('At least 3 slice points and 2 slices required')
    if not config['min_radius'] < config['max_radius']:
        raise ValueError('min_radius must be less than max_radius')
    if not 0 < config['bottom_percentile'] < config['top_percentile'] <= 100:
        raise ValueError('Invalid height percentiles')
    if config['min_inlier_ratio'] > 1 or config['min_arc_coverage'] > 2*np.pi:
        raise ValueError('Invalid inlier ratio or arc coverage')


def fit_trunk(points, candidate, association_radius=0.8, **tuning):
    config=dict(GEOMETRY_DEFAULTS,association_radius=association_radius)
    unknown=set(tuning)-set(config)
    if unknown: raise ValueError(f'Unknown geometry parameters: {unknown}')
    config.update(tuning)
    validate_geometry(config)
    g=config

    p=np.asarray(points,dtype=float)
    p=p[np.isfinite(p).all(axis=1)]
    p=p[np.linalg.norm(p[:,:2]-np.asarray(candidate[:2]),axis=1)<association_radius]
    if len(p)<g["min_points"]:return None
    bins=np.floor(p[:,2]/g["slice_height"]).astype(int)
    fits=[];rng=np.random.default_rng(7)
    for b in np.unique(bins):
        q=p[bins==b,:2]
        if len(q)<g["min_slice_points"]:continue
        best=None
        for _ in range(g["ransac_iterations"]):
            a=q[rng.choice(len(q),3,replace=False)]
            mat=2*(a[1:]-a[0]);rhs=(a[1:]**2).sum(axis=1)-(a[0]**2).sum()
            if abs(np.linalg.det(mat))<1e-5:continue
            center=np.linalg.solve(mat,rhs);radius=np.linalg.norm(a[0]-center)
            if not g["min_radius"]<=radius<=g["max_radius"] or np.linalg.norm(center-np.asarray(candidate[:2]))>g["max_center_offset"]:continue
            mask=np.abs(np.linalg.norm(q-center,axis=1)-radius)<g["circle_tolerance"]
            if mask.sum()<g["min_slice_points"] or mask.mean()<g["min_inlier_ratio"]:continue
            angle=np.sort(np.arctan2(*(q[mask]-center)[:,::-1].T))
            coverage=2*np.pi-np.diff(np.r_[angle,angle[0]+2*np.pi]).max()
            if coverage<g["min_arc_coverage"]:continue
            if best is None or mask.sum()>best[0]:best=(mask.sum(),center,radius)
        if best is not None:fits.append((b,*best))
    if len(fits)<g["min_slices"]:return None
    centers=np.array([f[2] for f in fits]);center=np.median(centers,axis=0)
    good=[f for f in fits if np.linalg.norm(f[2]-center)<g["slice_center_tolerance"]]
    if len(good)<g["min_slices"] or (max(f[0] for f in good)-min(f[0] for f in good))*g["slice_height"]<g["min_vertical_span"]:return None
    radius=float(np.median([f[3] for f in good]))
    surface=p[np.abs(np.linalg.norm(p[:,:2]-center,axis=1)-radius)<g["surface_tolerance"]]
    if len(surface)<3:return None
    bottom,top=np.percentile(surface[:,2],[g["bottom_percentile"],g["top_percentile"]]);height=top-bottom
    if height<g["min_height"]:return None
    return np.array([center[0],center[1],bottom,height,radius])

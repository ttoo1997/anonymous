'Canonical method names; legacy names are accepted only for compatibility.'
from copy import deepcopy

U2R_FULL_NAME = 'User-level Two-Round Regression'
METHODS = ('lasso', 'u2r', '2slr', 'sample_sdp', 'ldpiht')
DISPLAY_NAMES = {'u2r': 'U2R', '2slr': '2-SLR', 'sample_sdp': 'SampleSDP',
                 'ldpiht': 'LDPIHT', 'lasso': 'Lasso'}


def canonical(method):
    name = str(method).lower()
    return 'u2r' if name == 'proposed' else name


def display(method):
    name = canonical(method)
    return DISPLAY_NAMES.get(name, name)


def normalize_profiles(profiles):
    'Read old public profiles without changing the archived input file.'
    out = deepcopy(profiles)
    for dataset, methods in out.get('selected', {}).items():
        renamed = {}
        for method, parameters in methods.items():
            name = canonical(method)
            if name in renamed and renamed[name] != parameters:
                raise ValueError(f'Conflicting legacy/U2R profiles for {dataset}')
            renamed[name] = parameters
        out['selected'][dataset] = renamed
    for trial in out.get('trials', []):
        identity = trial.get('identity', {})
        if 'method' in identity:
            identity['method'] = canonical(identity['method'])
    return out

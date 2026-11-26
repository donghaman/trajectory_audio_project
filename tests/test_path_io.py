import csv
import os
import pytest
from path_io import read_trajectory


def test_unexpected_headers(tmp_path):
    p = tmp_path/'x.csv'
    p.write_text('x,y\n1,2\n', encoding='utf-8')
    with pytest.raises(ValueError, match='실제 열'):
        read_trajectory(str(p))


def test_valid_csv(tmp_path):
    p = tmp_path/'x.csv'
    with p.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=['frame','time_s','azimuth_deg','elevation_deg','distance_m','source'])
        w.writeheader(); w.writerow(dict(frame=0,time_s=0,azimuth_deg=0,elevation_deg=0,distance_m=1,source='A'))
    rows=read_trajectory(str(p))
    assert len(rows)==1


def test_project_sample_csv_files():
    project_root = os.path.dirname(os.path.dirname(__file__))
    sample_dir = os.path.join(project_root, 'Sample', '동선')
    for name in ('object_path_2.csv', 'object_path_3.csv', 'object_path_4.csv'):
        rows = read_trajectory(os.path.join(sample_dir, name))
        assert len(rows) == 361
        assert rows[0]['frame'] == 0
        assert rows[-1]['frame'] == 360

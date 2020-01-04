"""Container tests"""
import pandas as pd

import reda


def test_init():
    """test initializing an empty ERT container"""
    container = reda.ERT()


def test_init_with_data():
    """test initializing an ERT container and provide good data"""
    df = pd.DataFrame(
        [
            # normals
            (0, 1, 2, 4, 3, 1.1),
            (0, 1, 2, 5, 4, 1.2),
            (0, 1, 2, 6, 5, 1.3),
            (0, 1, 2, 7, 6, 1.4),
            (0, 2, 3, 5, 4, 1.5),
            (0, 2, 3, 6, 5, 1.6),
            (0, 2, 3, 7, 6, 1.7),
            (0, 3, 4, 6, 5, 1.8),
            (0, 3, 4, 7, 6, 1.9),
            (0, 4, 5, 7, 6, 2.0),
        ],
        columns=['timestep', 'a', 'b', 'm', 'n', 'r'],
    )
    container_good = reda.ERT(data=df)
    assert container_good.data.shape[0] == df.shape[0]

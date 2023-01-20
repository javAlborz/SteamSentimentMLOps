#from tests import _PATH_DATA
import os.path


@pytest.mark.skipif(not os.path.exists(file_path), reason="Data files not found")
def test_something_three():
    assert True

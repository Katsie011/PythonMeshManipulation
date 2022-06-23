"""
Little experiment to determine which feature extraction will give the best depth accuracy on the pydnet
"""


from ErrorEvaluationPydnetImgs import *

def test_orb():
    # TODO get orb features and check error at those points
    pass

def test_sift():
    # TODO get sift features and check error at those points
    pass


def test_moc():
    # TODO get moc features and check error at those points
    pass



if __name__ == "__main__":
    # import dataset

    """
    for each frame:
        - get dense-ified lidar
        - load predicted depth
        
        - check orb accuracy
        - check sift accuracy
        - check moc accuracy
        - check random point accuracy
    
    """
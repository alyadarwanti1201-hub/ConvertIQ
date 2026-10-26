import numpy as np

def create_features(data):
    data = data.copy()

    data['TotalPages'] = (
        data['Administrative']
        + data['Informational']
        + data['ProductRelated']
    )

    data['TotalDuration'] = (
        data['Administrative_Duration']
        + data['Informational_Duration']
        + data['ProductRelated_Duration']
    )

    data['ProductPageRatio'] = np.where(
        data['TotalPages'] > 0,
        data['ProductRelated'] / data['TotalPages'].replace(0, np.nan),
        0
    )

    data['AvgProductDuration'] = np.where(
        data['ProductRelated'] > 0,
        data['ProductRelated_Duration'] / data['ProductRelated'].replace(0, np.nan),
        0
    )

    return data

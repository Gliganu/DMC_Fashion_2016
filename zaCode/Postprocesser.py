def postprocessNAArticle(data):
    dataCopy = data.copy()
    dataCopy.loc[dataCopy['articleID'] == 'i1004001', 'prediction'] = 0
    return dataCopy


def postprocessZeroQuantArticle(data):
    dataCopy = data.copy()
    dataCopy.loc[dataCopy['quantity'] == 0, 'prediction'] = 0
    return dataCopy


def postprocessZeroPricedArticle(data):
    """
    for now, zero-priced articles have their prediction simply set to zero
    A possible extension is to check if the prediction for the order is overall tending to
    returning articles. If yes, then set the prediction 1
    """
    dataCopy = data.copy()
    dataCopy.loc[dataCopy['price'] == 0.0, 'prediction'] = 0
    return dataCopy

def postprocess(data, naArticle = True, zeroQuantity = True, zeroPrice = True):
    dataCopy = data.copy()

    if naArticle:
        dataCopy = postprocessNAArticle(dataCopy)
    if zeroQuantity:
        dataCopy = postprocessZeroQuantArticle(dataCopy)
    if zeroPrice:
        dataCopy = postprocessZeroPricedArticle(dataCopy)

    return dataCopy



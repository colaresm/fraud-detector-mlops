def get_params_by_prediction(data):
    sepal_length = data.get('sepal_length')
    sepal_width = data.get('sepal_width')
    petal_length = data.get('petal_length')
    petal_width = data.get('petal_width')
    return sepal_length,sepal_width,petal_length,petal_width

def get_params_to_train(data):
    hidden_layer_sizes = data.get('hidden_layer_sizes')
    max_iter = data.get('max_iter')

    return hidden_layer_sizes, max_iter


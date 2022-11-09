def repeat_and_reshape(tensor, repeat_n):
    n_scenes = tensor.shape[0]
    return (
        tensor.view(tensor.shape[0], 1, *tensor.shape[1:])
        .repeat(1, repeat_n, *([1] * (tensor.ndim - 1)))
        .view(n_scenes * repeat_n, *tensor.shape[1:])
    )


def repeat_and_reshape_all(tensor_list, repeat_n):
    out = []
    for tensor in tensor_list:
        out.append(repeat_and_reshape(tensor, repeat_n))
    return out

def torch_autodiff_grad_hess(
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    y_true: np.ndarray, y_pred: np.ndarray
):
    """Perform automatic differentiation to get the
    Gradient and the Hessian of `loss_function`."""
    y_true = torch.tensor(y_true, dtype=torch.float, requires_grad=False)
    y_pred = torch.tensor(y_pred, dtype=torch.float, requires_grad=True)
    loss_function_sum = lambda y_pred: loss_function(y_true, y_pred).sum()

    loss_function_sum(y_pred).backward()
    grad = y_pred.grad

    hess_matrix = torch.autograd.functional.hessian(loss_function_sum, y_pred, vectorize=True)
    hess = torch.diagonal(hess_matrix)

    return grad, hess

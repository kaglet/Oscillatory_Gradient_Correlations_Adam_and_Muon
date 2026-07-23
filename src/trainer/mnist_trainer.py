# TODO: Overwrite and replace this with trainer at higher level 
# TODO: See that model is passed because their lit module is not defined yet, we already have defined upper level configs, and this should be upper level even if a custom build.
def train_and_eval(
    model: torch.nn.Module,
    # TODO: Change this so knowledge of lightning module not required and only lightning module instance built already is passed in as only requirement to a Trainer
    optim: Partial[Optimizer],
    params: list[dict[str, Any]],
    dataloader: Type[DataLoader],
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    training_domain: torch.Tensor,
    lit_module: Type[MNISTLitModule],
    trainer: pl.Trainer,
):


    lit = lit_module(
        model=model,
        optim=optim,
        params=params,
        dataloader=dataloader,
        target_fn=target_fn,
        training_domain=training_domain,
    )

    # train
    trainer.fit(lit)

    # evaluate the model over the domain to assess the fit
    final_eval = lit(training_domain.reshape(-1, 1))
    final_eval = final_eval.detach().cpu().numpy().ravel()

    # return the final evaluation of our model:
    # a shape-(N,) numpy-array
    return final_eval
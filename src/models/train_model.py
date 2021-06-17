def train(trainer, div2k, model):
    trainer.fit(model=model, datamodule=div2k)


def test(trainer, div2k, model):
    trainer.test(model=model, datamodule=div2k)

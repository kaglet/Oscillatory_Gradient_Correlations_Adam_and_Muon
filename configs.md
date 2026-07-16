## [Trainer](https://hydra.cc/docs/advanced/instantiate_objects/overview/) 

This can be used in a python program itself which is better. Just pass a config object, especially if the arguments of a field are a config anyway. Config object can be a normal dict, or their custom Dict types it seems. 
At the overview you would have something like this if the same architecture is used I believe or the parameters to train on are borrowed from the architecture itself. Observe difference between setting key-value pairs in a dict, vs passing an actual expected object. 

You can set it as a string or the named key. It still identifies the correct thing I think either way. See 2 examples below, where learning rate is set differently. 

```python
with target_whitelist("my_app.*"):
    opt = instantiate(
        cfg.optimizer,
        lr=0.2,
    )
print(opt)
# Optimizer(algo=SGD,lr=0.2)
```

```python
with target_whitelist("my_app.*"):
    trainer = instantiate(
        cfg.trainer,
        optimizer={"lr": 0.3},
        dataset={"name": "cifar10", "path": "/datasets/cifar10"},
    )
    print(trainer)
    # Trainer(
    #  optimizer=Optimizer(algo=SGD,lr=0.01),
    #  dataset=Dataset(name=Imagenet, path=/datasets/imagenet)
    # )
```


```yml
trainer:
  _target_: my_app.Trainer
  optimizer:
    _target_: my_app.Optimizer
    algo: SGD
    lr: 0.01
  dataset:
    _target_: my_app.Dataset
    name: Imagenet
    path: /datasets/imagenet
```
To be clear when we have a config like the above. The actual object is sourced in the yaml. When we define this nested structure, what is needed to replace it, the correct value for the key, is a dictionary. That dictionary may have value literals where required. The key value pair can literally be a dictionary i.e. a config. You can literally swap out a config, or an actual value, or redefine your own quickly. I'm sure rather you can let hydra handle swapping out the config. I.e. sourcing the correct config object. I just need to figure this out. #question 

## [Recursive instantiation](https://hydra.cc/docs/advanced/instantiate_objects/overview/#disable-recursive-instantiation)

Without recursive instantiation, you get that the Trainer object receives an OmgegaConf DictConfig for the dataset and optimiser instead of the insanstiated objects. Not sure why you would use this?

## [Parameter conversion strategies](https://hydra.cc/docs/advanced/instantiate_objects/overview/#parameter-conversion-strategies)

Learn OmegaConf containers (DictConfig, ListConfig), compared to primitive dicts and lists. Learn about duck typing, dataclass, attrs classes, variable interpolation and custom resolvers again. OmgeConf.to_object. Structured Config. target_whitelist and when exactly to use. 

I do not yet understand when to use OmegaConf containers (DictConfig, ListConfig) to pass through. I think hydra itself, uses omega conf to define its yaml into this type. Then we have the convenient attribute access for keys, apparently also "duck typing", variable interpolation, and custom resolvers. Links given, forgot what last 2 are, first link is hard to understand #todo. 

Define different terms, Structured Configs - python data class, DictConfigs - hydra custom class, OmegaConf containers. All are different types. 

I also do not well understand *when* to use conversion strategy partial. Nor what behind the scenes stuff is at play. 

It is also not properly defined what convert means, what the default is with a link to say OmegaConf. Can search, but is frustrating for newcomer, this type is not linked to its doc #todo #question. 
Convert seems to mean if any DictConfig, is found, whether to convert it to a normal dict or not. 

I wonder how Hydra can enforce types like Hydra Zen does/can? #question 

OmegaConf from yaml has all nested dicts as being OmegaConf containers. Sometimes that is bad because when OmegaConf takes ownership of dicts when they are a *value*, sure that can  be useful later on to do OmegaConf things. Sometimes it consumes a dict declaration whether implicit or explicit. And you get to tell it if you want it to treat even user defined dicts as OmegaConf containers. 

In the example given, when the data class instance object is printed, it's representation is a Structured Config. 
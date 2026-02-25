# Plugins Quickstart

`retrain` now supports a unified plugin workflow across swappable parts:

1. create one file
2. set one TOML key
3. run training

## 60-second transform plugin

Generate a scaffold:

```bash
retrain init-plugin --kind transform --name my_transform --with-test
```

This creates `plugins/my_transform.py` and an optional smoke test.

Use it in TOML:

```toml
[algorithm]
transform_mode = "plugins.my_transform.my_transform"

[algorithm.transform_params]
scale = 1.0
```

## Advantage plugin

```bash
retrain init-plugin --kind advantage --name my_adv
```

```toml
[algorithm]
advantage_mode = "plugins.my_adv.my_adv"

[algorithm.advantage_params]
scale = 2.0
```

## Full algorithm plugin

If you want to override the whole advantage pipeline:

```bash
retrain init-plugin --kind algorithm --name my_algo
```

```toml
[algorithm]
algorithm_mode = "plugins.my_algo.my_algo"

[algorithm.params]
alpha = 0.1
```

When `algorithm_mode` is set, it takes priority over `advantage_mode` + `transform_mode`.

## Discover what is available

```bash
retrain plugins
retrain plugins --json
```

## Plugin search paths

Default search path:

```toml
[plugins]
search_paths = ["plugins"]
strict = true
```

`strict = true` fails fast on plugin load/shape errors.

from marllib import marl

env = marl.make_env(environment_name="disaster", map_name="test1")
algo = marl.algos.mappo(hyperparam_source="disaster")

model = marl.build_model(env, algo, {
    "core_arch": "mlp",
    "encode_layer": "64-64"
})

results = algo.fit(
    env, model,
    stop={"timesteps_total": 1e7},  # 正确生效
    num_gpus=1,
    num_workers=10,
    share_policy="all",
    local_mode=False,
    checkpoint_freq=1000,
    checkpoint_end=True,
)

ALPHA_PARAM_SCHEMAS = {
    "alpha_075": [
        ("window", int),
        ("window_corr_vwap", int),
        ("window_corr_volume", int),
    ],
    "alpha_full_factor_094": [
        ("window", int),
        ("window_corr_vwap", int),
        ("window_corr_volume", int),
    ],
    "alpha_full_factor_034":[
        ("window", int),
        ("factor", float),
        ("window_corr_vwap", int),
    ],
    "alpha_full_factor_099_eff_macd":[
        ("fast", int),
        ("slow", int),
        ("window_norm", int),
    ]
}
GEN_SCHEMAS = {
    "1_1": {
        "gen": [("freq", int), ("threshold", float), ("halflife", float)],
        "params": [("window", int), ("factor", float)]
    },
    "1_2": {
        "gen": [("freq", int), ("upper", float), ("lower", float)],
        "params": [("window", int), ("factor", float)]
    },
    "1_3": {
        "gen": [("freq", int), ("score", int), ("entry", float), ("exit", float)],
        "params": [("window", int), ("factor", float)]
    },
    "1_4": {
        "gen": [("freq", int), ("entry", float), ("exit", float), ("smooth", float)],
        "params": [("window", int), ("factor", float)]
    }
}
def parse_by_schema(values, schema):
    result = {}
    for i, (key, cast) in enumerate(schema):
        if i < len(values):
            result[key] = cast(values[i])
    return result
def get_param_schema(gen, alpha_name):
    # ưu tiên schema theo alpha nếu có
    if alpha_name in ALPHA_PARAM_SCHEMAS:
        return ALPHA_PARAM_SCHEMAS[alpha_name]

    # fallback về default của gen
    return GEN_SCHEMAS[gen]["params"]
def parse_alpha_config(gen, config, alpha_name):
    parts = config.split("_")
    schema = GEN_SCHEMAS[gen]

    # parse gen params
    gen_values = parse_by_schema(parts, schema["gen"])
    rest = parts[len(schema["gen"]):]

    # lấy param schema
    param_schema = get_param_schema(gen, alpha_name)

    params = parse_by_schema(rest, param_schema)

    # bỏ freq khỏi gen_params
    gen_params = {k: v for k, v in gen_values.items() if k != "freq"}

    return gen_values["freq"], gen_params, params

    
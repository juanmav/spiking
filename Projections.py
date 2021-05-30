# Layer projections
# Retina => Parrot layer
conn_retina_parrot_dict = {
    "connection_type": "convergent",
    "mask": {
        "grid": {
            "rows": 1,
            "columns": 1
        }
    }
}

# Parrot layer => V1
conn_parrot_v1_dict = {
    "connection_type": "convergent",
    "mask": {
        "circular": {
            "radius": 0.5
        }
    },
    # "kernel": {
    #    "gaussian": {
    #        "p_center": 1.0,
    #        "sigma": 0.05
    #    }
    # },
    "weights": 3.0,
    # "delays": {
    #     "linear": {
    #         "c": 0.1,
    #         "a": 0.2
    #     }
    # }
}

# p(d) = c + ad
# V1 interconnections
conn_ee_dict = {
    "connection_type": "convergent",
    "mask": {
        "circular": {
            "radius": 1.0
        }
    },
    "kernel": {
        "gaussian": {
            "p_center": 1.0,
            "sigma": 0.15
        }
    },
    "weights": 1.0,
    # "delays": {
    #     "linear": {
    #         "c": 0.1,
    #         "a": 0.2
    #     }
    # }
}

conn_ie_dict = {
    "connection_type": "convergent",
    "mask": {
        "circular": {
            "radius": 1.0
        }
    },
    "kernel": {
        "gaussian": {
            "p_center": 1.0,
            "sigma": 0.15
        }
    },
    "weights": -2.0,
    # "delays": {
    #     "linear": {
    #         "c": 0.1,
    #         "a": 0.2
    #     }
    # }
}

conn_ei_dict = {
    "connection_type": "convergent",
    "mask": {
        "circular": {
            "radius": 1.0
        }
    },
    "kernel": {
        "gaussian": {
            "p_center": 1.0,
            "sigma": 0.15
        }
    },
    "weights": 1.0,
    # "delays": {
    #     "linear": {
    #         "c": 0.1,
    #         "a": 0.2
    #     }
    # }
}

conn_ii_dict = {
    "connection_type": "convergent",
    "mask": {
        "circular": {
            "radius": 1.0
        }
    },
    "kernel": {
        "gaussian": {
            "p_center": 1.0,
            "sigma": 0.15
        }
    },
    "weights": -0.5,
    # "delays": {
    #     "linear": {
    #         "c": 0.1,
    #         "a": 0.2
    #     }
    # }
}

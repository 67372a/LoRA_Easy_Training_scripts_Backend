from pathlib import Path


def validate(args: dict) -> tuple[bool, bool, list[str], dict, dict]:
    over_errors = []
    if "args" not in args:
        over_errors.append("args is not present")
    if "dataset" not in args:
        over_errors.append("dataset is not present")
    if over_errors:
        return False, False, over_errors, {}, {}
    args_pass, args_errors, args_data = validate_args(args["args"])
    dataset_pass, dataset_errors, dataset_data = validate_dataset_args(args["dataset"])
    over_pass = args_pass and dataset_pass
    over_errors = args_errors + dataset_errors
    tag_data = {}
    if not over_errors:
        validate_restarts(args_data, dataset_data)
        validate_warmup_ratio(args_data, dataset_data)
        validate_rex(args_data, dataset_data)
        tag_data = validate_save_tags(dataset_data)
        validate_existing_files(args_data)
    sdxl = validate_sdxl(args_data)
    if not over_pass:
        return False, sdxl, over_errors, args_data, dataset_data, tag_data
    return True, sdxl, over_errors, args_data, dataset_data, tag_data


def validate_args(args: dict) -> tuple[bool, list[str], dict]:
    # sourcery skip: low-code-quality
    passed_validation = True
    errors = []
    output_args = {}

    for key, value in args.items():
        if not value:
            passed_validation = False
            errors.append(f"No data filled in for {key}")
            continue
        if "fa" in value and value["fa"]:
            output_args["network_module"] = "networks.lora_fa"
            del value["fa"]
        for arg, val in value.items():
            if arg == "network_args":
                vals = []
                for k, v in val.items():
                    if k == "algo":
                        output_args["network_module"] = "lycoris.kohya"
                    elif k == "unit":
                        output_args["network_module"] = "networks.dylora"
                    if k in [
                        "down_lr_weight",
                        "up_lr_weight",
                        "block_dims",
                        "block_alphas",
                        "conv_block_dims",
                        "conv_block_alphas",
                    ]:
                        for i in range(len(v)):
                            v[i] = str(v[i])
                        vals.append(f"{k}={','.join(v)}")
                        continue
                    if k == "preset" and v == "":
                        continue
                    vals.append(f"{k}={v}")
                val = vals
            if arg == "optimizer_args":
                vals = []
                for k, v in val.items():
                    if v in ["true", "false"]:
                        v = v.capitalize()
                    vals.append(f"{k}={v}")
                val = vals
            if arg == "lr_scheduler_args":
                vals = [f"{k}={v}" for k, v in val.items()]
                val = vals
            if arg == "keep_tokens_separator" and len(val) < 1:
                passed_validation = False
                errors.append("Keep Tokens Separator is an empty string")
                continue
            if not val:
                continue
            output_args[arg] = val
        if "fa" in value:
            del value["fa"]

    file_inputs = [
        {"name": "pretrained_model_name_or_path", "required": True},
        {"name": "output_dir", "required": True},
        {"name": "sample_prompts", "required": False},
        {"name": "logging_dir", "required": False},
    ]

    for file in file_inputs:
        if file["required"] and file["name"] not in output_args:
            passed_validation = False
            errors.append(f"{file['name']} is not found")
            continue
        if file["name"] in output_args and not Path(output_args[file["name"]]).exists():
            passed_validation = False
            errors.append(
                f"{file['name']} input '{output_args[file['name']]}' does not exist"
            )
            continue
        elif file["name"] in output_args:
            output_args[file["name"]] = Path(output_args[file["name"]]).as_posix()
    if "network_module" not in output_args:
        output_args["network_module"] = "networks.lora"
    return passed_validation, errors, output_args


def validate_dataset_args(args: dict) -> tuple[bool, list[str], dict]:
    passed_validation = True
    errors = []
    output_args = {"general": {}, "subsets": []}

    for key, value in args.items():
        if not value:
            passed_validation = False
            errors.append(f"No Data filled in for {key}")
            continue
        if key == "subsets":
            continue
        for arg, val in value.items():
            if not val:
                continue
            if arg == "max_token_length" and val == 75:
                continue
            output_args["general"][arg] = val

    for item in args["subsets"]:
        sub_res = validate_subset(item)
        if not sub_res[0]:
            passed_validation = False
            errors += sub_res[1]
            continue
        output_args["subsets"].append(sub_res[2])
    return passed_validation, errors, output_args


def validate_subset(args: dict) -> tuple[bool, list[str], dict]:
    passed_validation = True
    errors = []
    output_args = {key: value for key, value in args.items() if value}
    name = "subset"
    if "name" in output_args:
        name = output_args["name"]
        del output_args["name"]
    if "image_dir" not in output_args or not Path(output_args["image_dir"]).exists():
        passed_validation = False
        errors.append(f"Image directory path for '{name}' does not exist")
    else:
        output_args["image_dir"] = Path(output_args["image_dir"]).as_posix()
    return passed_validation, errors, output_args


def validate_restarts(args: dict, dataset: dict) -> None:
    if "lr_scheduler_num_cycles" not in args:
        return
    if "lr_scheduler_type" not in args:
        return
    if "max_train_steps" in args:
        steps = args["max_train_steps"]
    else:
        steps = calculate_steps(
            dataset["subsets"],
            args["max_train_epochs"],
            dataset["general"]["batch_size"]
            * args.get("gradient_accumulation_steps", 1),
        )
    steps = steps // args["lr_scheduler_num_cycles"]
    args["lr_scheduler_args"].append(f"first_cycle_steps={steps}")
    del args["lr_scheduler_num_cycles"]


def validate_warmup_ratio(args: dict, dataset: dict) -> None:
    if "warmup_ratio" not in args:
        return
    if "max_train_steps" in args:
        steps = args["max_train_steps"]
    else:
        steps = calculate_steps(
            dataset["subsets"],
            args["max_train_epochs"],
            dataset["general"]["batch_size"]
            * args.get("gradient_accumulation_steps", 1),
        )
    steps = round(steps * args["warmup_ratio"])
    if "lr_scheduler_type" in args:
        args["lr_scheduler_args"].append(
            f"warmup_steps={steps // args.get('lr_scheduler_num_cycles', 1)}"
        )
    else:
        args["lr_warmup_steps"] = steps
    del args["warmup_ratio"]


def validate_rex(args: dict, dataset: dict) -> None:
    if "lr_scheduler_type" not in args:
        return
    if args["lr_scheduler_type"].split(".")[-1] != "Rex":
        return
    if "max_train_steps" in args:
        steps = args["max_train_steps"]
    else:
        steps = calculate_steps(
            dataset["subsets"],
            args["max_train_epochs"],
            dataset["general"]["batch_size"]
            * args.get("gradient_accumulation_steps", 1),
        )
    args["lr_scheduler_args"].append(f"total_steps={steps}")


def validate_existing_files(args: dict) -> None:
    file_name = Path(
        f"{args['output_dir']}/{args.get('output_name', 'last')}.safetensors"
    )
    offset = 1
    while file_name.exists():
        file_name = Path(
            f"{args['output_dir']}/{args.get('output_name', 'last')}_{offset}.safetensors"
        )
        offset += 1
    if offset > 1:
        print(f"Duplicate file found, changing file name to {file_name.stem}")
        args["output_name"] = file_name.stem


def validate_sdxl(args: dict) -> bool:
    if "sdxl" not in args:
        return False
    del args["sdxl"]
    return True


def validate_save_tags(dataset: dict) -> dict:
    tags = {}
    for subset in dataset["subsets"]:
        subset_dir = Path(subset["image_dir"])
        if not subset_dir.is_dir():
            continue
        for file in subset_dir.iterdir():
            if not file.is_file():
                continue
            if file.suffix != subset["caption_extension"]:
                continue
            get_tags_from_file(subset_dir.joinpath(file.name), tags)
    return dict(sorted(tags.items(), key=lambda item: item[1], reverse=True))


def get_tags_from_file(file: str, tags: dict) -> None:
    with open(file, "r") as f:
        temp = f.read().replace(", ", ",").split(",")
        for tag in temp:
            if tag in tags:
                tags[tag] += 1
            else:
                tags[tag] = 1


def calculate_steps(subsets: list, epochs: int, batch_size: int) -> int:
    steps = 0
    for subset in subsets:
        image_count = 0
        files = list(Path(subset["image_dir"]).iterdir())
        for file in files:
            if file.suffix.lower() not in {
                ".png",
                ".bmp",
                ".gif",
                ".jpeg",
                ".jpg",
                ".webp",
            }:
                continue
            image_count += 1
        steps += image_count * subset["num_repeats"]
    steps = (steps * epochs) // batch_size
    return steps

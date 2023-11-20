def modify_image(image, filter_name):
    # Convert spaces to underscores
    filter_module_name = filter_name.replace(" ", "_")
    filter_module_name = "src.app.filters." + filter_module_name

    try:
        # Dynamically import the specified filter module
        filter_module = __import__(filter_module_name, fromlist=['modify_image'])

        # Call the modify_image function from the filter module
        result_image = filter_module.modify_image(image)

        return result_image
    except ImportError:
        print(f"Filter '{filter_module_name}' not found.")
        return None

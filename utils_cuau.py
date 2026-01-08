
# Map from index to atomic number
number_to_element: dict[int, int] = {
    0: 29,  # Cu
    1: 79,  # Au
    2: 1,  # mask
}
element_to_number: dict[int, int] = {v: k for k, v in number_to_element.items()}
num_elements: int = len(number_to_element)

# Mask index for discrete flow
mask_index: int = len(number_to_element) - 1

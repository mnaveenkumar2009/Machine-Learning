def apply_conv_to_image(conv,original_image)
    final=


hori_line_cov=[[1, 1],
                [-1,-1]]
vertical_line_conv = [[1, -1], 
                      [1, -1]]
conv_list = [vertical_line_conv]

original_image = load_my_image()
print("Original image")
show(original_image)
for conv in conv_list:
    filtered_image = apply_conv_to_image(conv, original_image)
    print("Output: ")
    show(filtered_image)
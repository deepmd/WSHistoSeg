def get_path(image_id, num_layers):
    row = image_id + ','
    for layer in range(1, num_layers+1):
        filename = image_id.split('/')[1].replace('.bmp', f'_layer{layer}_cam.npy')
        layer_cam = image_id.split('/')[0] + f'/CAMs/Layer{layer}/' + filename
        row = row + layer_cam + ','
    row += '\n'
    return row


if __name__ == "__main__":
    input_path = "/home/reza/Documents/WSHistoSeg/datasets/folds/GLAS/fold-0/test/image_ids.txt"
    out_path = "/home/reza/Documents/WSHistoSeg/datasets/folds/GLAS/fold-0/test/image_cams.txt"

    infile = open(input_path, 'r')
    outfile = open(out_path, "a")
    input_files = infile.readlines()
    for input_file in input_files:
        image_id = input_file.strip('\n')
        row = get_path(image_id, 4)
        outfile.write(row)
    infile.close()
    outfile.close()

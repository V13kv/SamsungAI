import torch

from torch.autograd import Variable


def calculateLocalContrastL1(img, window_size):
    """Get contrast level in terms of Manhattan metrics for each difference between internal window of image and shifted image, place them (differences) to the 4-th dimension"""
    img = img.permute(0, 2, 3, 1)   # (B, H, W, C) {from (batch_size, channels, height, width) -> (batch, height, width, channels)}
    x_diff = img[:, window_size:256 - window_size, window_size:256 - window_size]   # get internal window (formed via substitution window_size from borders)
    x = x_diff

    # Iterate over each partition of window_size in img
    for i in range(-window_size, window_size + 1):
        for j in range(-window_size, window_size + 1):
            if i == -window_size and j == -window_size:
                # Compute difference between internal window and moved img to the left (moved to the window_size amount)
                img_diff = x_diff - img[:, 0:256 - 2 * window_size, 0:256 - 2 * window_size]    # preserve batch_size, shift height and width to window_size to the left
                img_diff = torch.sum(torch.abs(img_diff), 3)    # sum absolute values over dim=3 (i.e. sum all channels together, their absolute values)
                img_diff = torch.unsqueeze(img_diff, 3) # insert new singleton at dim=3 (i.e. channels dimension)
                x = img_diff
            # Don't compute diffference when shifted window is exactly at position of internal window (will get 0 anyway, no deposit)
            elif (i == 0) and (j == 0):
                continue
            else:
                # Compute difference between internal window and moved img to the left (moved to the window_size amount)
                img_diff = x_diff - img[:, window_size + i:256 - window_size + i, window_size + j:256 - window_size + j]
                img_diff = torch.sum(torch.abs(img_diff), 3)     # sum absolute values over dim=3 (i.e. sum all channels together, their absolute values)
                img_diff = torch.unsqueeze(img_diff, 3) # insert new singleton at dim=3 (i.e. channels dim)
                x = torch.cat((x, img_diff), 3) # concatenate previous img_diff and current calculated img_diff on dim=3 (channels dim)
    nrand = np.array([i for i in range(120)])
    trand = torch.from_numpy(nrand).type(torch.long)

    # Return only 120 (see above) difference layers, i.e. shape (batch_size, height, width, diff_layers_cnt), where height and width is height and width OF INTERNAL WINDOW!!!
    return abs(x[:, :, :, trand])


def calculateGlobalContrastL1(img1, img2, points_number = 5):
    """Calculate global contrast difference between images (randomly choose points of points_number number and calculate L1` metrics for them )"""
    # Exchange some dimensions of images for convenience
    img1 = img1.permute(0, 2, 3, 1)   # (B, H, W, C), i.e. (batch, height, width, channels)}
    img2 = img2.permute(0, 2, 3, 1) # same for the second image

    height, width = img1.shape[1], img1.shape[2]

    select_points = torch.tensor(np.zeros( (img1.size(0), 1, points_number, 1) )) # (batch_size, 1, points_number, 1)
    # print(select_points.shape)

    # Generate random (uniform distribution applied) indices to calculate contrast on
    rnd_height_points = torch.randint(0, height, (points_number,))
    rnd_width_points = torch.randint(0, width, (points_number,))
    rnd_height_points_1 = torch.randint(0, height, (points_number,))
    rnd_width_points_2= torch.randint(0, width, (points_number,))

    # Choose random points on the first image
    img_points1 = img1[:, rnd_height_points, rnd_width_points, :]    # (batch_size, random_height_points, random_width_points, channels)
    img_points2 = img1[:, rnd_height_points_1, rnd_width_points_2, :]

    # Count contrast difference between these random points
    img1_diff = (img_points1 - img_points2)
    img1_diff = torch.sum(torch.abs(img1_diff), 2)

    # Choose random points on the second image
    img2_points1 = img2[:, rnd_height_points, rnd_width_points,:]
    img2_points2 = img2[:, rnd_height_points_1, rnd_width_points_2, :]

    # Count contrast difference between these random points
    img2_diff = (img2_points1 - img2_points2)
    img2_diff = torch.sum(torch.abs(img2_diff), 2)

    return img1_diff, img2_diff


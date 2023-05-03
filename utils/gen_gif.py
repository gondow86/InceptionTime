from PIL import Image

pictures_center = []
pictures_cross = []
pictures_cos = []

# for epoch in range(1, 301):
#     for itr in range(16):
#         file_name = "%03d_%02d_Fmap.png" % (epoch, itr)
#         file_path = "./figure/feature_map_center/"
#         img = Image.open(file_path + file_name)
#         pictures.append(img)

for epoch in range(1, 301):
    file_name = "%03d_01_Fmap.png" % (epoch)
    file_path = "./figure/feature_map_center/"
    img = Image.open(file_path + file_name)
    pictures_center.append(img)

for epoch in range(1, 301):
    file_name = "%03d_01_Fmap.png" % (epoch)
    file_path = "./figure/feature_map/"
    img = Image.open(file_path + file_name)
    pictures_cross.append(img)

for epoch in range(1, 301):
    file_name = "%03d_01_Fmap.png" % (epoch)
    file_path = "./figure/feature_map_cos/"
    img = Image.open(file_path + file_name)
    pictures_cos.append(img)

pictures_center[0].save(
    "center_loss.gif",
    save_all=True,
    append_images=pictures_center[1:],
    optimize=False,
    duration=200,
    loop=0,
)

pictures_cross[0].save(
    "cross_entropy.gif",
    save_all=True,
    append_images=pictures_cross[1:],
    optimize=False,
    duration=200,
    loop=0,
)

pictures_cos[0].save(
    "cos_loss.gif",
    save_all=True,
    append_images=pictures_cos[1:],
    optimize=False,
    duration=200,
    loop=0,
)

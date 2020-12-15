class AdvancedHairAugmentation:
    def __init__(self, hairs: int = 4, hairs_folder: str = ""):
        self.hairs = hairs
        self.hairs_folder = hairs_folder

    def __call__(self, img):
        n_hairs = random.randint(0, self.hairs)

        if not n_hairs:
            return img

        height, width, _ = img.shape  # target image width and height
        hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]

        for _ in range(n_hairs):
            hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(hair_images)))
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            dst = cv2.add(img_bg, hair_fg)
            img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst

        return img


from albumentations.core.transforms_interface import ImageOnlyTransform

class AddBackgroud(ImageOnlyTransform):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, background_img_paths, p = 0.5):
        
        super(AddBackgroud, self).__init__()
        
        self.background_img_paths = background_img_paths
        self.back_img_path = str(random.choice(self.background_img_paths))
        self.p = p
        
    def  apply(self, image, **params):
        
        if np.random.rand() >= self.p:
            return image
        
        y1=random.randint(10,110)
        y2=y1+32
        x1=random.randint(10,261)
        x2=x1+32
        
        prev_bk = image[0][0].sum()
        image_bk = cv2.imread(self.back_img_path)[y1:y2, x1:x2]
        
        for y in range(32):
            for x in range(32):
                if image[y][x].sum() != prev_bk:
                    image_bk[y][x] = image[y][x]
        return image_bk
import numpy as np
import scipy as sp
import scipy.signal  # one option for a 2D convolution library
import cv2
from numba import jit
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from bayespy.nodes import GaussianARD, Gamma, Gaussian, Dirichlet, Dot, Categorical, Wishart, Mixture
from bayespy.inference import VB
import bayespy.plot as bpplt
from scipy.signal import convolve2d, convolve
import scipy.io
import glob, os


class DeBlur:
    def __init__(self, blurry_img, blur_kernel_upper_bound, blur_kernel_orientation, video_bool):
        self.blur_kernel_upper_bound = blur_kernel_upper_bound
        self.blur_kernel_orientation = blur_kernel_orientation
        self.gamma = 2.2

        if not video_bool:
            self.blurry_img = cv2.imread(blurry_img)
            user_patch = cv2.selectROI(self.blurry_img)
            self.blur_patch = self.blurry_img[int(user_patch[1]):int(user_patch[1] + user_patch[3]), int(user_patch[0]):int(user_patch[0] + user_patch[2])]
            cv2.imwrite('patch.png', self.blur_patch)
            self.blurry_gray_patch_gamma_corrected = self.pre_process()

            out_img = self.execution_main()
            cv2.imwrite('out_img.png', out_img)

        else:
            video_frames = self.pre_process_video(blurry_img)
            self.blurry_img = cv2.imread(video_frames[0])
            user_patch = cv2.selectROI(self.blurry_img)
            coordinates = [int(user_patch[1]), int(user_patch[1] + user_patch[3]),
                           int(user_patch[0]), int(user_patch[0] + user_patch[2])]

            output_frames = []
            for index, frame in enumerate(video_frames):
                self.blurry_img = cv2.imread(frame)
                self.blur_patch = self.blurry_img[coordinates[0]:coordinates[1], coordinates[2]:coordinates[3]]
                self.blur_kernel_upper_bound = blur_kernel_upper_bound
                self.blur_kernel_orientation = blur_kernel_orientation
                self.gamma = 2.2
                self.blurry_gray_patch_gamma_corrected = self.pre_process()

                dir = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/output_frames/'
                outimg = self.execution_main()
                cv2.imwrite(dir + 'frame{}.png'.format(index), outimg)
                output_frames.append(dir + 'frame{}.png'.format(index))
                print('completed frame {} out of {}'.format(index, len(video_frames)))
            os.system('convert -loop 0 %s output.gif' % ' '.join(output_frames))

    def pre_process_video(self, video_path):
        video_object = cv2.VideoCapture(video_path)
        img_count, render = 0, 1

        while render:
            render, img = video_object.read()
            cv2.imwrite('video_frames/frame{}.jpg'.format(img_count), img)
            img_count += 1

        frame_nums = []
        dir = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/video_frames/'
        for frame in glob.glob(dir + '*.jpg'):
            frame_num = frame.split('/')[-1].split('frame')[1].split('.jpg')[0]
            frame_nums.append(int(frame_num))

        frame_nums = sorted(frame_nums)

        final_frames = []
        for index, frame_num in enumerate(frame_nums):
            frame_name = dir + 'frame' + str(frame_num) + '.jpg'
            final_frames.append(frame_name)
        final_frames = final_frames[:-1]
        return final_frames

    def remove_gamma_correction(self, image):
        out_image = np.ndarray((image.shape[0], image.shape[1]), dtype=image.dtype)
        for row in np.arange(0, image.shape[0]):
            for column in np.arange(0, image.shape[1]):
                out_image[row, column] = 255 * ((image[row, column] / 255) ** (1.0 / self.gamma))
        cv2.imwrite('patch3.png', out_image)
        return out_image

    def add_gamma_correction(self, image):
        out_image = np.ndarray((image.shape[0], image.shape[1], image.shape[2]), dtype=image.dtype)
        count = 0
        for channel in np.arange(0, image.shape[2]):
            for row in np.arange(0, image.shape[0]):
                for column in np.arange(0, image.shape[1]):
                    out_image[row, column, channel] = 255 * ((image[row, column, channel] / 255) ** self.gamma)
                    print('completed {} out of {}'.format(count, image.shape[2] * image.shape[1]*image.shape[0]))
                    count += 1
        return out_image

    def compute_gradient(self, patch):
        patch_x_gradient = np.absolute(cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=5))
        cv2.imwrite('grad_x.png', patch_x_gradient)
        patch_y_gradient = np.absolute(cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=5))
        cv2.imwrite('grad_y.png', patch_y_gradient)
        concat = np.concatenate((patch_x_gradient, patch_y_gradient))
        cv2.imwrite('concat.png', concat)
        return concat

    def pre_process(self):
        # blurry_gray = cv2.cvtColor(self.blurry_img, cv2.COLOR_BGR2GRAY)
        blurry_gray_patch = cv2.cvtColor(self.blur_patch, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('patch2.png', blurry_gray_patch)
        # blurry_gray_gamma_corrected = self.remove_gamma_correction(blurry_gray)
        return self.remove_gamma_correction(blurry_gray_patch)

    # def subsample(self):
    #     kernel_height, kernel_width = self.kernel.shape
    #
    #     subsample_img = np.zeros((int(np.ceil(self.kernel.shape[0])), int(np.ceil(self.kernel.shape[1]))), dtype=np.dtype('float'))
    #     row_bool = False
    #     column_bool = False
    #     subsample_x = 0
    #     subsample_y = 0
    #     for y in np.arange(0, output_img.shape[0]):
    #         if not row_bool:
    #             for x in np.arange(0, output_img.shape[1]):
    #                 if not column_bool:
    #                     column_bool = True
    #                     subsample_img[subsample_y, subsample_x] = output_img[y, x]
    #                     subsample_x += 1
    #                 else:
    #                     column_bool = False
    #
    #         subsample_x = 0
    #         if not row_bool:
    #             row_bool = True
    #         else:
    #             row_bool = False
    #             column_bool = False
    #             subsample_y += 1


    def run_variational_bayes_inference(self, patch_gradient, kernel, latent_gradient, prior_kernel, prior_latent):
        new_latent = latent_gradient
        new_kernel = kernel
        np.random.seed(1)
        convolve_sample = convolve2d(latent_gradient, kernel, 'same')

        # L_p = GaussianARD(0, prior_latent['gamma'][0], plates=(latent_gradient.shape[0] * latent_gradient.shape[1],), name='L_p')
        K = GaussianARD(kernel, prior_kernel['lambda'], plates=(kernel.shape[0], kernel.shape[1]), name='K')
        P = GaussianARD(convolve_sample, 1, plates=(convolve_sample.shape[0], convolve_sample.shape[1]), name='P')

        # L_p = Dot(prior_latent['pi'][0], L_p)
        # K = Dot(prior_kernel['pi'], K)
        # F = Dot(L_p, P)
        patch_gradient = cv2.resize(patch_gradient, (int(convolve_sample.shape[1]), int(convolve_sample.shape[0])))
        tau = Gamma(1e-3, 1e-3, name='tau')
        Y = GaussianARD(P, tau, name='Y')
        Y.observe(patch_gradient)
        Q = VB(Y, P, tau)

        #Q.optimize(P, riemannian=False, method='gradient', maxiter=5)
        Q.update(repeat=100, plot=True, tol=5e-3)
        # bpplt.pyplot.plot()
        # bpplt.pdf(P, np.linspace(1e-6, 70, num=convolve_sample.shape[0] * convolve_sample.shape[1]), color='k', name=r'P')
        # bpplt.pyplot.tight_layout()
        # bpplt.pyplot.show()
        lower_bound = Q.compute_lowerbound()
        #print('MAX: {}'.format(np.amax(P.g)))
        factor = np.amax(P.g)

        # uncomment this
        # Z = GaussianARD(K, tau, name='Z')
        # Z.observe(kernel)
        # R = VB(Z, K, tau)
        # R.update(repeat=100, plot=True, tol=5e-3)

        for i in np.arange(0, P.g.shape[0]):
            for j in np.arange(0, P.g.shape[1]):
                value = float(str(P.g[i, j]).replace(',', '.'))
                if value < 0:
                    new_latent[i, j] = -1 * value
                    if new_latent[i, j] > 255:
                        new_latent[i, j] = 255
                else:
                    new_latent[i, j] = 0

        #uncomment this
        # for i in np.arange(0, K.g.shape[0]):
        #     for j in np.arange(0, K.g.shape[1]):
        #         value = float(str(K.g[i, j]).replace(',', '.'))
        #         if value < 0.0:
        #             new_kernel[i, j] = 0
        #         elif value > 255.:
        #             new_kernel[i, j] = 255
        #         else:
        #             new_kernel[i, j] = value

        # distribution = np.zeros(shape=(kernel.shape), dtype=np.float64)
        # for m in np.arange(0, kernel.shape[0]):
        #     for n in np.arange(0, kernel.shape[1]):
        #         distribution[m, n] += kernel[m, n]
        #
        # for i in np.arange(0, patch_gradient.shape[0]):
        #     for j in np.arange(0, patch_gradient.shape[1]):
        #         distribution[i, j] += patch_gradient[i, j]





        #
        # N = patch_gradient.shape[0]
        # D = patch_gradient.shape[1]
        # K= 10
        #
        # alpha = Dirichlet(1e-5*np.ones(K),
        #                   name='alpha')
        # Z = Categorical(alpha, plates=(N,),
        #                 name='Z')
        # mu = Gaussian(np.zeros(D), 1e-5*np.identity(D), plates=(K,), name='mu')
        # Lambda = Wishart(D, 1e-5*np.identity(D), plates=(K,), name='Lambda')
        #
        # Y = Mixture(Z, Gaussian, mu, Lambda, name='Y')
        # Z.initialize_from_random()
        # Q = VB(Y, mu, Lambda, Z, alpha)
        #
        # Y.observe(patch_gradient)
        # Q.update(repeat=1000)
        #
        # bpplt.gaussian_mixture_2d(Y, alpha=alpha, scale=2)
        # bpplt.pyplot.show()


        # D = 3
        # X = GaussianARD(0, 1, shape=(D,), plates=(1,patch_gradient.shape[1]), name='X')
        # alpha = Gamma(1e-3, 1e-3, plates=(D,), name='alpha')
        # C = GaussianARD(0, alpha, shape=(D,), plates=(patch_gradient.shape[0],1), name='C')
        # F = Dot(C, X)
        # tau = Gamma(1e-3, 1e-3, name='tau')
        # Y = GaussianARD(F, tau, name='Y')
        # Y.observe(patch_gradient)
        #
        # Q = VB(Y, C, X, alpha, tau)
        # Q.update(repeat=20)
        # bpplt.pyplot.figure()
        # bpplt.pdf(Q['tau'], np.linspace(60, 140, num=100))
        # bpplt.pyplot.show()






        # alpha = Gamma(1e-3, 1e-3, plates=(3,), name='alpha')
        #
        # L_p = GaussianARD(0, alpha=alpha, shape=(3,), plates=(patch_gradient.shape[0], patch_gradient.shape[1]), name='Lp')
        #
        # K = GaussianARD(1, 1e4, shape=(3,), plates=(patch_gradient.shape[0], patch_gradient.shape[1]), name='K')
        #
        # F = Dot(L_p, K)
        #
        # tau = Gamma(1e-3, 1e-3, name='tau')
        #
        # Y = GaussianARD(F, tau, name='Y')
        #
        # Y.observe(patch_gradient)
        #
        # Q = VB(Y, L_p, K, alpha, tau)
        #
        # Q.update(repeat=20, tol=5e-3)
        #
        # # bpplt.pyplot.subplot(2, 1, 1)
        # # bpplt.pdf(Q['K'], np.linspace(-10, 20, num=100), color='k', name=r'\K')
        # bpplt.pyplot.subplot(2, 1, 2)
        # bpplt.pdf(Y, np.linspace(1e-6, 0.08, num=100), color='k', name=r'\L_p')
        # bpplt.pyplot.tight_layout()
        # bpplt.pyplot.show()
        # cv2.imshow('latent', new_latent)
        # cv2.waitKey(0)
        return kernel, new_latent

    def modified_richardson_lucy(self, kernel, iterations):
        stack = []
        for channel in cv2.split(self.blurry_img):
            channel = np.float64(convolve2d(channel, kernel, 'same'))
            channel += 0.1 * channel.std() * np.random.standard_normal(channel.shape)

            img = np.full(channel.shape, 0.5)
            kernel_mirror = kernel[::-1, ::-1]

            for index in range(iterations):
                relative_blur = channel / convolve(img, kernel, mode='same')
                img *= convolve(relative_blur, kernel_mirror, mode='same')
                print('iteration {}'.format(index))

            stack.append(img)
        output_img = np.uint8(cv2.merge((stack[0], stack[1], stack[2])))
        return output_img

    def kernel_threshold(self, kernel):
        threshold = np.max(kernel) / 15
        for row in np.arange(0, kernel.shape[0]):
            for column in np.arange(0, kernel.shape[1]):
                if kernel[row, column] < threshold:
                    kernel[row, column] = 0
        return kernel

    def load_priors(self):
        prior_latent_mat = scipy.io.loadmat(
            '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/priors/linear_street_4.mat')  # weird formatting

        prior_latent = np.array([
            {'pi': prior_latent_mat['priors'][0][0][0][0],
             'gamma': prior_latent_mat['priors'][0][0][1][0]},
            {'pi': prior_latent_mat['priors'][0][1][0][0],
             'gamma': prior_latent_mat['priors'][0][1][1][0]},
            {'pi': prior_latent_mat['priors'][0][2][0][0],
             'gamma': prior_latent_mat['priors'][0][2][1][0]},
            {'pi': prior_latent_mat['priors'][0][3][0][0],
             'gamma': prior_latent_mat['priors'][0][3][1][0]},
            {'pi': prior_latent_mat['priors'][0][4][0][0],
             'gamma': prior_latent_mat['priors'][0][4][1][0]},
            {'pi': prior_latent_mat['priors'][0][5][0][0],
             'gamma': prior_latent_mat['priors'][0][5][1][0]},
            {'pi': prior_latent_mat['priors'][0][6][0][0],
             'gamma': prior_latent_mat['priors'][0][6][1][0]},
            {'pi': prior_latent_mat['priors'][0][7][0][0],
             'gamma': prior_latent_mat['priors'][0][7][1][0]}])

        prior_kernel = {
            'pi': 1,
            'lambda': 1
        }
        return prior_latent, prior_kernel

    def post_process(self, out_img):
        out_img = self.add_gamma_correction(out_img)

        y, cr, cb = cv2.split(cv2.cvtColor(out_img, cv2.COLOR_BGR2YCrCb))
        y = cv2.equalizeHist(y)

        merged = cv2.merge((y, cr, cb))
        post_processed_img = cv2.cvtColor(merged, cv2.COLOR_YCR_CB2BGR)

        return post_processed_img

    def execution_main(self):
        kernel_list = []
        latent_list = []
        patch_subsampled_list = []
        patch_gradient = self.compute_gradient(self.blurry_gray_patch_gamma_corrected)
        scales = math.ceil(-2 * math.log((3 / self.blur_kernel_upper_bound), 2.0))
        prior_latent, prior_kernel = self.load_priors()

        for scale in np.arange(scales):
            resize_factor = (1 / math.sqrt(2)) ** (scales - scale + 1)
            patch_gradient_resized = cv2.resize(patch_gradient, (int(patch_gradient.shape[1] * resize_factor), int(patch_gradient.shape[0] * resize_factor)), interpolation=cv2.INTER_LINEAR)

            if scale == 0:
                kernel = np.array([[0, 0, 0], [1., 1., 1.], [0, 0, 0]]) / 9.
                latent_gradient = patch_gradient_resized
                if self.blur_kernel_orientation == 'vertical':
                    kernel = kernel.transpose()
            else:
                kernel = cv2.resize(kernel_list[scale-1], (int(kernel_list[scale-1].shape[0] * math.sqrt(2)), int(kernel_list[scale-1].shape[1] * math.sqrt(2))))
                latent_gradient = cv2.resize(latent_list[scale-1], (int(latent_list[scale-1].shape[1] * math.sqrt(2)), int(latent_list[scale-1].shape[0] * math.sqrt(2))))

            # Run Inference
            kernel, latent_gradient = self.run_variational_bayes_inference(patch_gradient_resized, kernel, latent_gradient, prior_kernel, prior_latent[scale])

            res = cv2.resize(self.blurry_gray_patch_gamma_corrected, None, fx=kernel.shape[1]/self.blur_patch.shape[1], fy=kernel.shape[0]/self.blur_patch.shape[0], interpolation=cv2.INTER_CUBIC)

            kernel_list.append(kernel)
            latent_list.append(latent_gradient)
            patch_subsampled_list.append(res)

        # for count, kernel in enumerate(latent_list):
        #     cv2.imwrite('latent{}.png'.format(count), kernel)
        # for count, kernel in enumerate(patch_subsampled_list):
        #     cv2.imwrite('patchhhh{}.png'.format(count), kernel)
        # cv2.imshow('patch{}'.format(1), patch_subsampled_list[-1])
        # cv2.waitKey(0)

        out_img = self.modified_richardson_lucy(kernel=self.kernel_threshold(kernel_list[2]), iterations=10)
        cv2.imwrite('RL-img.png', out_img)
        out_img = self.post_process(out_img)

        return out_img





if __name__ == "__main__":
    # user input params
    # blurry_image = "/Users/ms621y/Desktop/GaTech/assignments/Final_Project/deblur_images/test1/fountain_blurry.png"
    # blur_patch = "/Users/ms621y/Desktop/GaTech/assignments/Final_Project/deblur_images/test1/fountain_im_kernel.png"
    #blurry_image = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/deblur_images/test2/ian1.jpg'
    #blur_patch = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/deblur_images/test2/ian1_blurry_closeup.jpg'
    #blurry_image = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/video_blur.mp4'

    #########
    # test 1
    # blurry_image = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/FinalProj_demo/test1/test1.png'

    # test 2
    # blurry_image = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/FinalProj_demo/test2/test2.png'

    # test 3
    # blurry_image = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/FinalProj_demo/test3/test3.png'

    # test 4
    #blurry_image = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/FinalProj_demo/test4/test4.png'

    # test 5
    #blurry_image = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/FinalProj_demo/test5/test5.png'

    # test 6
    #blurry_image = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/FinalProj_demo/test6/test6.png'

    # # test 7
    # blurry_image = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/FinalProj_demo/test7/test7.png'

    # test 8
    # blurry_image = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/FinalProj_demo/test8/test8.png'

    # test 9
    # blurry_image = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/FinalProj_demo/test9/test9.png'

    #test 10 video
    # blurry_image = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/FinalProj_demo/test10/test10.mp4'

    # test 11 video
    # blurry_image = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/FinalProj_demo/test11/test11.mp4'

    # blurry_image = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/FinalProj_demo/test1/test1.png'

    # paper test 1
    #blurry_image = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/deblur_images/test1/fountain_blurry.png'
    #blurry_image = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/deblur_images/test2/ian1_blurry_closeup.jpg'

    #blurry_image = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/deblur_images/test3/lyndsey2_blurry.jpg'

    # blurry_image = '/Users/ms621y/Desktop/GaTech/assignments/Final_Project/deblur_images/test4/klette1_blurry.png'
    #
    # blur_kernel_upper_bound = 40               # 8 scales
    # blur_kernel_orientation = 'horizontal'
    # video_bool = False

    blurry_image = sys.argv[1]
    blur_kernel_upper_bound = sys.argv[2]
    blur_kernel_orientation = sys.argv[3]
    video_bool = sys.argv[4]

    DeBlur(blurry_image, blur_kernel_upper_bound, blur_kernel_orientation, video_bool)

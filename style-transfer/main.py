import argparse
import distutils.util
import tensorflow as tf
import src.multi_style_transfer as mst
import time


def main():
    parser = argparse.ArgumentParser()

# Train ############################################################################################################
    parser.add_argument("-f", "--flag", type=distutils.util.strtobool, default='True')
    parser.add_argument("-gn", "--gpu_number", type=int, default=0)
    # parser.add_argument("-p", "--project", type=str, default=r"E:\毕设\STYLE_TRANSFER\onedrive\style-transfer\result\multi-result\MST")
    parser.add_argument("-p", "--project", type=str, default="st/style-transfer/result/MST")

    ## Train images
    # parser.add_argument("-ctd", "--content_dataset", type=str, default=r"E:\毕设\STYLE_TRANSFER\workspace\style-transfer\data\mock-coco\train2014")
    parser.add_argument("-ctd", "--content_dataset", type=str, default="st/style-transfer/data/val2017")
    parser.add_argument("-cts", "--content_data_size", type=int, default=256)
    # parser.add_argument("-sti", "--style_image", type=str, default=r"E:\毕设\STYLE_TRANSFER\workspace\style-transfer\data\style-images\0_udnie.jpg")
    parser.add_argument("-sti", "--style_image", type=str, default="st/style-transfer/images/style_crop/0_udnie.jpg")

    ## Train Iteration
    parser.add_argument("-n",  "--niter", type=int, default=2)
    parser.add_argument("-ns", "--nsnapshot", type=int, default=1000)
    parser.add_argument("-mx", "--max_to_keep", type=int, default=10)

    ## Train Parameter
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-a", "--alpha", type=float, default=1.0)
    parser.add_argument("-m", "--momentum", type=float, default=0.9)
    parser.add_argument("-m2", "--momentum2", type=float, default=0.999)

    ## loss weight
    parser.add_argument("-lc", "--content_loss_weights", type=float, default=1.5e0)
    parser.add_argument("-ls", "--style_loss_weights", type=float, default=1e2)
    parser.add_argument("-lt", "--tv_loss_weight", type=float, default=2e2)

# Test #############################################################################################################
    # parser.add_argument("-tsd", "--test_dataset", type=str, default=r"E:\毕设\STYLE_TRANSFER\workspace\style-transfer\data\test")
    parser.add_argument("-tsd", "--test_dataset", type=str, default="st/style-transfer/images/test")
    parser.add_argument("-scw", "--style_control_weights", type=float, nargs=16)

    args = parser.parse_args()
    gpu_number = args.gpu_number
    train_flag = args.flag

    with tf.device('/gpu:{}'.format(gpu_number)):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        with tf.Session(config=config) as sess:
            ## Make Model
            model = mst.mst(args, sess)
            start = time.time()
            ## TRAIN / TEST
            if train_flag:
                model.train(train_flag)
                end = time.time()
                print("train time:%.3f s"%(end-start))
            else:
                model.test(train_flag)
                end = time.time()
                print("test time:%.3f s"%(end-start))


if __name__ == '__main__':
    main()


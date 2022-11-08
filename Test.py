import torch
from Utils import *
from Model import Network
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from scipy.stats import spearmanr as SROCC

def test_model():

    device = 'cuda:0'
    valset_dir = './Win5_160x160/'
    load_all_model_path = './PreTrainedModels/'
    test_scene_len = 2
    scene_num = 10

    net = Network().to(device)
    cudnn.benchmark = True

    all_model = os.listdir(load_all_model_path)
    label_list = np.zeros([test_scene_len * 22, len(all_model)])
    data_list = np.zeros([test_scene_len * 22, len(all_model)])
    val_SRCC_all = []

    test_scene_id_list = []
    for a in range(scene_num):
        for b in range(a+1,scene_num):
            test_scene_id_list.append([a,b])

    new_all_model = []
    for list_id in test_scene_id_list:
        for id, model_name in enumerate(all_model):
            test_scene_id = [int(model_name[0]),int(model_name[2])]
            if test_scene_id == list_id:
                new_all_model.append(model_name)

    for id, model_name in enumerate(new_all_model):
        load_model_path = load_all_model_path + '/' + str(model_name) + '/DeeBLiF_epoch70.pth.tar'
        model = torch.load(load_model_path, map_location={'cuda:0': device})
        net.load_state_dict(model['state_dict'])
        net.eval()
        index = 0
        scene_list = ['greek', 'Flowers', 'Sphynx', 'museum', 'rosemary',
                      'Vespa', 'Swans_1', 'dishes', 'Palais_du_Luxembourg', 'Bikes']
        test_scene_id = [int(model_name[0]),int(model_name[2])]
        for test_scene in test_scene_id:
            image_path = valset_dir + '/' + scene_list[test_scene]
            image_list = os.listdir(image_path)
            for test_image in image_list:
                patch_path = valset_dir + '/' + scene_list[test_scene] + '/' + test_image
                patch_list = os.listdir(patch_path)
                output_list = 0
                for val_patch in patch_list:
                    each_patch_path = patch_path + '/' + val_patch
                    with h5py.File(each_patch_path, 'r') as hf:
                        label = np.array(hf.get('score_label'))
                        data = np.array(hf.get('data'))
                        data = data / 255
                        data = np.expand_dims(data, axis=0)
                        data = np.expand_dims(data, axis=0)
                        data = torch.from_numpy(data.copy())
                        data = Variable(data).to(device)
                    with torch.no_grad():
                        out_score = net(data)
                    output_list += out_score.cpu().numpy()
                label_list[index, id] = label.item()
                data_list[index, id] = output_list.item() / len(patch_list)
                index += 1

        val_SRCC = SROCC(data_list[:,id], label_list[:,id]).correlation
        val_SRCC_all.append(val_SRCC)
        print(test_scene_id)
        print('SROCC :----    %f' % val_SRCC)
    print('Average SROCC :----   %f' % np.mean(val_SRCC_all))

    # save in h5 file and test in matlab
    f = h5py.File('DeeBLiF_result_Win5.h5', 'w')
    f.create_dataset('predict_data', data=data_list)
    f.create_dataset('score_label', data=label_list)
    f.close()

if __name__ == '__main__':
    test_model()

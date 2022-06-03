from model.pointnet2_rot_dir import PointCloudNormRotDir
from model.pointnet2_rot_dir_double import PointCloudNormRotDirDouble
from model.pointnet_rot_dir import PointNetRotDir
from model.pointnet_rot_dir_double import PointNetRotDirDouble


def get_model(model_name):
    if model_name == 'point2_rot_dir':
        model = PointCloudNormRotDir()
    elif model_name == 'point2_rot_dir_double':
        model = PointCloudNormRotDirDouble()
    elif model_name == 'pointnet_rot_dir':
        model = PointNetRotDir()
    elif model_name == 'pointnet_rot_dir_double':
        model = PointNetRotDirDouble()
    else:
        raise NotImplemented('Wrong Model')
    return model



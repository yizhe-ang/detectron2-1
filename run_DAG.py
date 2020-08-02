from detectron2.config import get_cfg
from detectron2_1.adv import DAGAttacker
from detectron2 import model_zoo


img_path = 'data/samples/WechatIMG18.png'
cfg_path = 'output/rcnn_2/config.yaml'
weights_path = 'output/rcnn_2_resume/model_0004999.pth'

# cfg = get_cfg()
# cfg.merge_from_file(cfg_path)
# cfg.MODEL.WEIGHTS = weights_path

print('Preparing config file...')
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
# Not sure why have to do this
cfg.DATALOADER.NUM_WORKERS = 0

print('Initializing attacker...')
attacker = DAGAttacker(cfg)

print('Start the attack...')
coco_instances_results = attacker.run_DAG(vis=False)

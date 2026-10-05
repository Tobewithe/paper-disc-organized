"""Read installed official model/loss/assigner source for a read-only diagnostic."""
import inspect
from ultralytics import YOLO
from ultralytics.utils.loss import v8SegmentationLoss,v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner
from frozen_mechanism_probe import ROOT

model=YOLO(str(ROOT/'weights/yolo26m-seg.pt')).model
print('MODEL',type(model).__name__,'END2END',model.end2end,'ARGS',model.args)
for name,obj in [('SEG_INIT',v8SegmentationLoss.__init__),('SEG_LOSS',v8SegmentationLoss.loss),
    ('SEG_CALC',v8SegmentationLoss.calculate_segmentation_loss),('SINGLE_MASK',v8SegmentationLoss.single_mask_loss),
    ('ASSIGN_LOSS',v8DetectionLoss.get_assigned_targets_and_loss),('ASSIGN_FORWARD',TaskAlignedAssigner._forward),
    ('HEAD_FORWARD',type(model.model[-1]).forward)]:
    print('\nSOURCE',name,inspect.getfile(obj));print(inspect.getsource(obj))

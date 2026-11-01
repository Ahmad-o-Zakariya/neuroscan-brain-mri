import pytest
np=pytest.importorskip("numpy")
pytest.importorskip("sklearn");pytest.importorskip("matplotlib")
from scripts.evaluate import _plot_classifier,_plot_segmentation

def test_evaluation_creates_confusion_roc_and_dice_iou_figures(tmp_path):
 y=np.array([0,1,2,3]);p=np.eye(4)*.85+.15/4
 _plot_classifier(y,y,p,tmp_path);_plot_segmentation([.2,.6,.8],[.1,.4,.7],tmp_path)
 assert {"classifier_confusion_matrix.png","classifier_roc.png","segmenter_dice_iou.png"} <= {x.name for x in tmp_path.iterdir()}

def test_gated_end_to_end_counts_skipped_localization(tmp_path):
 torch=pytest.importorskip("torch")
 from PIL import Image
 from scripts.evaluate import _evaluate_gated_cases
 class EmptySegmenter:
  def __call__(self,x): return torch.full((len(x),1,x.shape[-2],x.shape[-1]),-10.)
 (tmp_path/"glioma").mkdir()
 cls=[];segs=[]
 for i in range(2):
  image=tmp_path/"glioma"/f"glioma_{i}.png";mask=tmp_path/f"glioma_{i}_mask.png";Image.new("L",(16,16),i).save(image);Image.new("L",(16,16),0).save(mask)
  cls.append({"path":str(image)});segs.append({"image":str(image),"mask":str(mask)})
 result=_evaluate_gated_cases(cls,np.array([[.6,.2,.1,.1],[.1,.1,.1,.7]]),segs,.5,EmptySegmenter(),"cpu",16,.5)
 assert result["localization_not_run_after_gate"]==1
 assert result["localization_run_and_scored"]==1
 assert result["n_paired_cases_scored"]==2
 assert result["tumour_screening_sensitivity"]==.5

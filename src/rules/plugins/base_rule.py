class BaseViolationRule:
    """
    Base class for all traffic violation rules.
    Teammates should inherit from this class to create new rules (e.g., HelmetDetection).
    """
    def __init__(self):
        self.name = "Base Rule"

    def evaluate(self, detections, **kwargs):
        """
        Evaluates the detections and returns a list of violations.
        Override this in subclasses.
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

    def _calculate_iou(self, boxA, boxB):
        """Utility for calculating Intersection over Union (IoU)"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

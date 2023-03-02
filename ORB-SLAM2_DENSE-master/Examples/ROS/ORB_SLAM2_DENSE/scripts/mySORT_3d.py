import numpy as np


class BoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    self.box = bbox
    # 匹配到了的时候会在tracker.update()中置0，没有匹配的话每帧在predictz()中+1
    self.time_since_update = 0
    self.id = BoxTracker.count
    # 每创建一个tracker,count + 1
    BoxTracker.count += 1
    # 有匹配update时会把history清空，没遇到匹配时会在predict中将上次的状态压入
    self.history = []
    # hits在匹配到了之后update()时+1 表示匹配到了几次
    self.hits = 0
    # hit_streak在匹配到了之后update()时+1 表示连续匹配到了多少次 在predict时会把上次没更新的tracker的hit_streak置0
    self.hit_streak = 0
    # 在predict时age+1 表示自从创建这个tracker已经存在了多久（几帧）
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    只有匹配到了新的match后才update
    有新的detection跟之前的tracker匹配了之后，更新这个tracker
    更新了的tracker history会清空，留着添加这次的detection参数
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    # 更新box
    self.box = bbox

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    每一帧都对所有的trackers进行predict
    """
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(self.box.reshape((1,9)))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return self.box.reshape((1, 9))


def iou_batch(dets, trks):
    """
    input： bb_test:dets      bb_gt:trks   [[x,y,z,l,w,h,yaw,label,score],[x,y,z,l,w,h,yaw,label,score],...]
    需要将输入格式转为xy平面投影的[x1,y1,x2,y2,score]格式
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    输出的iou矩阵 行对应trks,列对应dets
    """
    bb_test = np.array([dets[..., 0] - dets[..., 3]/2, dets[..., 1] - dets[..., 4]/2, dets[..., 0] + dets[..., 3]/2, dets[..., 1] + dets[..., 4]/2, dets[..., 8]])
    bb_test = np.transpose(bb_test)
    bb_gt = np.array([trks[..., 0] - trks[..., 3]/2, trks[..., 1] - trks[..., 4]/2, trks[..., 0] + trks[..., 3]/2, trks[..., 1] + trks[..., 4]/2, trks[..., 8]])
    bb_gt = np.transpose(bb_gt)

    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    a = bb_test[..., 0]
    b = bb_gt[..., 0]
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))




def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    # 如果上一帧没有trackers，matches返回0行2列的空array， unmatched_detections返回的array包含所有detections的下标,
    # unmatched_trackers返回0行8列空array；
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,9),dtype=int)

  # 计算iou矩阵
  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    # 相当于matrix中只有> threshold的元素被置为1，剩下为0
    a = (iou_matrix > iou_threshold).astype(np.int32)
    # 如果正好有一对一的匹配成功(各行各列只有一个大于阈值的IOU) 则直接返回匹配成功的下标 [det1 last1], [det2 last2]
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  # 将未匹配的detections和trackers加入list
  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  #matches 和 matched_indices有什么区别？不是一样都大于iou_threshold吗
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=4, min_hits=2, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 8))):
    """
    Params:
      dets - a numpy array of detections in the format [[x,y,z,l,w,h,yaw,label,score],[x,y,z,l,w,h,yaw,label,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 8)) for frames without detections).
    Returns a similar array, where the last column is the object ID.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 9))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6], pos[6], 0]
      # 判断是否存在nan值，存在的话在to_del[]队列中加入下标等待删除
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)
    # matched:[[0 1], [1 2]]每一行为[det trks]的索引
    # unmatched_dets, unmatched_trks: [] 分别为dets或trks的索引


    # update matched trackers with assigned detections
    # 匹配到的trackers，去对应的trackers中更新新的观测
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        # 对于没有匹配到的detection，认为是新发现的，将坐标作为输入创建tracker
        trk = BoxTracker(dets[i,:])
        # 将这个tracker加入到trackers队列中
        self.trackers.append(trk)
    i = len(self.trackers)
    # 将trackers列表反转倒着取
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        # 如果满足这些条件 将tracker的id加入ret(return)列表返回
        #  #######################################################################   需要改进 因为这里不会输出没有检测到的tracker
        # 如果该tracker是当前帧匹配到的，并且： 连续匹配了几帧或者目前还没运行那么多帧

        # if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
        #     #学习一下这里的元素拼接
        #     ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive

        # 除此之外 还要输出虽然当前没有匹配但已经稳定的tracker
        if (trk.hits >= self.min_hits or trk.age <= self.min_hits):
            ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        # 如果该tracker已经max_age帧没有匹配并且它的hit匹配次数没有超过min_hits次，就删除
        if(trk.time_since_update >= self.max_age and trk.hits < self.min_hits):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,10))

# 测试
trackers = []
iou_threshold = 0.13
new_trks = np.array([[10, 10, 30, 3, 2, 1,0, 1, 0.5],
                 [4, 1, 6, 3, 2, 1, 0, 1, 0.5]])
trks = np.array([[0, 0, 0, 2, 2, 2, 0, 1, 0.5],
                 [1, 1, 3, 3, 2, 1, 0, 1, 0.5],
                 [4, 1, 6, 3, 2, 1, 0, 1, 0.5]])
dets = np.array([[1, 1, 3, 3, 2, 1, 0, 1, 0.5],
                 [4, 1, 6, 3, 2, 1, 0, 1, 0.5]])
new_dets = np.array([[0, 0, 0, 2, 2, 2, 0, 1, 0.5]])
mot_tracker = Sort()
track_bbs_ids = mot_tracker.update(new_trks)
track_bbs_ids = mot_tracker.update(trks)
track_bbs_ids = mot_tracker.update(dets)
track_bbs_ids = mot_tracker.update(new_dets)
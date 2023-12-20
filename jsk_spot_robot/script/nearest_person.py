import rospy
import tf2_ros
import tf2_geometry_msgs
from jsk_recognition_msgs.msg import BoundingBoxArray
from geometry_msgs.msg import PoseStamped, TransformStamped, PoseArray, Pose
import numpy as np
import time
import threading

import pyglet

window = pyglet.window.Window(800, 600, caption="estimated pos")
offset_x = 700
offset_y = 300
batch = pyglet.graphics.Batch()
spot_square = pyglet.shapes.Rectangle(offset_x, offset_y, 100, 50, color=(180, 132,0), batch=batch)
spot_square.anchor_position=(50,25)

@window.event
def on_draw():
    window.clear()
    batch.draw()

viz_thread = threading.Thread(target=lambda:pyglet.app.run())
viz_thread.start()


class PIRegulator:
    def __init__(self, p_gain, i_gain, num):
        self.p_gain = p_gain
        self.i_gain = i_gain
        # 減衰率
        self.k = 0.7

        self.buf = np.zeros(num)
        self.time_buf = np.ones(num)*time.time()
        self.weight = np.zeros(num)

    def get(self, delta):
        get_time = time.time()
        # bufを一つずらす
        self.buf[1:] = self.buf[0:-1]
        self.buf[0] = delta
        self.time_buf[1:] = self.buf[0:-1]
        self.time_buf[0] = get_time
        # 重みの計算(最近の時間のものほど信頼する)
        self.weight = np.exp(-self.k*(get_time - self.time_buf))
        self.weight = self.weight/np.sum(self.weight)
        # 重みをかけて和を取る
        sum_buf = np.dot(self.buf, self.weight)
        ret = self.p_gain*delta + self.i_gain*sum_buf
        print(f"delta:{delta}, sum:{sum_buf}, ret:{ret}")
        return ret

class KalmanFilter:
    def __init__(self, cov_init, Q, R):
        self.is_state_initialized = False
        self.cov = cov_init
        self.A:np.ndarray = np.eye(4)
        self.C:np.ndarray = np.block([np.eye(2), np.eye(2)*0.0])
        self.Q = Q
        self.R = R
        self.last_updated_time = time.time()


    def initState(self, init_state):
        if (not self.is_state_initialized):
            self.is_state_initialized = True
            self.state = init_state
        else:
            pass

    def update(self, x_obs, y_obs):
        updated_time = time.time()
        delta_t = min(updated_time - self.last_updated_time, 0.5)
        identity = np.eye(2)
        zero = np.zeros((2,2))
        self.A :np.ndarray = np.block([[identity, identity*delta_t], [zero, identity]])
        self.last_updated_time = updated_time
        self.state = self.A@self.state
        self.cov = self.A*self.cov * self.A.T + self.Q

        self.obs = np.array([[x_obs], [y_obs]])
        self.obs_pred = self.C @ self.state
        self.delta_obs = (self.obs - self.obs_pred)
        self.cov_obs = self.C @ self.cov @ self.C.T + self.R
        self.kalman_gain = self.cov@self.C.T@np.linalg.inv((self.cov_obs + self.C@self.cov@self.C.T))
        self.state = self.state + self.kalman_gain@self.delta_obs
        self.cov = (np.eye(4) - self.kalman_gain@self.C)@self.cov
        print(f"estimated_state:{self.state}")
        print(f"estimated_cov:{self.cov}")



        
class RelativeTfConverter:
    def __init__(self):
        rospy.init_node("rel_tf_node")
        #tfを保存するためのtf_buffer
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(100))
        #tfをsubscribeするためのtf_listener
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # tfをpublishするためのtf_buffer
        self.broadcaster = tf2_ros.TransformBroadcaster()
        # bbox_arrayをsubscribe
        rospy.Subscriber("/spot_recognition/bbox_array", BoundingBoxArray, self.callback)
        # /nearest_personというtopicに追従したい人のtfをpublishする
        self.pub = rospy.Publisher("/nearest_person", PoseStamped, queue_size=10)

        # x方向のPIレギュレータ
        self.x_PI_regulator = PIRegulator(1.0, 0.0, 5)
        # y方向のPIレギュレータ
        self.y_PI_regulator = PIRegulator(1.0, 0.0, 5)

        # 最後に認識結果が送られてきた時刻
        self.last_updated_time = time.time()

        # 認識した中で一番追従候補らしいもの
        self.nearest_pose_stamped_from_spot = PoseStamped()
        self.nearest_pose_stamped_from_spot.pose.position.x=0
        self.nearest_pose_stamped_from_spot.pose.position.y=0

        # 一つ前の司令位置
        self.pre_dest_pos_from_odom = PoseStamped()
        self.pre_dest_pos_from_odom.pose.position.x=0
        self.pre_dest_pos_from_odom.pose.position.y=0

        # 初期の分散共分散行列
        init_cov = np.diag([0.3, 0.3, 0.1, 0.1])
        # 更新プロセスでの共分散行列
        Q = np.diag([0.04, 0.04, 0.001, 0.001])
        # 観測プロセスでの共分散行列
        R = np.diag([0.03, 0.03])
        # カルマンフィルタ
        self.kalman_filter = KalmanFilter(init_cov, Q, R)
        # 5Hzで回るはず?
        self.rate = rospy.Rate(5)

        # 描画
        self.estimated_circle = pyglet.shapes.Circle(offset_x,offset_y,10, color=(0,255,0, 125), batch=batch)
        self.observed_circle = pyglet.shapes.Circle(offset_x, offset_y, 10, color=(255,0,0, 125), batch=batch)
        self.center_axis_x = pyglet.shapes.Line(offset_x, offset_y, offset_x+50, offset_y, color=(255, 0, 0), batch=batch)
        self.center_axis_y = pyglet.shapes.Line(offset_x, offset_y, offset_x, offset_y+50, color=(0, 255, 0), batch=batch)
        self.vel = pyglet.shapes.Line(offset_x, offset_y, offset_x, offset_y, color=(0,0,255), batch=batch)
        self.viz_scale = 50
        rospy.spin()

    def cvtBox2PoseStamped(self, box):
        # BoundingBoxArrayをPoseStampedに変換する関数
        original_pose_stamped = PoseStamped()
        original_pose_stamped.header = box.header
        original_pose_stamped.pose = box.pose
        return original_pose_stamped

    
    def callback(self, msg:BoundingBoxArray):
        self.delta_t = time.time() - self.last_updated_time
        self.last_updated_time = time.time()
        # msgの座標系のid(原点が73B2から見て約-26m下にある)
        from_id = "odom"
        # 変換したい座標系のid(原点はspotのbody中心)
        target_id = "body"
        # from->targetへのtfの変換を表す変数
        transform_to_body = self.tf_buffer.lookup_transform(target_id, from_id, rospy.Time(0), rospy.Duration(1.0))
        transform_to_odom = self.tf_buffer.lookup_transform(from_id, target_id, rospy.Time(0), rospy.Duration(1.0))
        
        if len(msg.boxes) > 0:
            self.nearest_pose_stamped_from_spot = PoseStamped()
            self.nearest_distance = 1e10
            self.updated = False
            for i, box in enumerate(msg.boxes):
                # msg.boxesに含まれるboxを一つづつPoseStampedに変換する
                pose_stamped = self.cvtBox2PoseStamped(box)
                # 座標をspot原点に変換する
                pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform_to_body)
                # spotからの距離
                distance = ((pose_transformed.pose.position.x - self.nearest_pose_stamped_from_spot.pose.position.x)**2 + (pose_transformed.pose.position.y - self.nearest_pose_stamped_from_spot.pose.position.y)**2)**0.5
                # print(f"id={i}, pos={pose_transformed.pose.position}")
                if (distance < self.nearest_distance and pose_transformed.pose.position.x < 0 and abs(pose_transformed.pose.position.y) < 4):
                    # x座標が負かつy方向に4m以内の人で距離が最も近い人を追従対象にする
                    self.nearest_distance = distance
                    self.nearest_pose_stamped_from_spot= pose_transformed
                    self.nearest_pose_stamped_from_odom = pose_stamped
                    self.updated = True

            
            if self.updated:
                # alpha = (self.nearest_distance-1)/self.nearest_distance
                if not self.kalman_filter.is_state_initialized:
                    init_state = np.array([[self.nearest_pose_stamped_from_odom.pose.position.x], [self.nearest_pose_stamped_from_odom.pose.position.y], [0], [0]])
                    self.kalman_filter.initState(init_state)
                else:
                    self.kalman_filter.update(self.nearest_pose_stamped_from_odom.pose.position.x, self.nearest_pose_stamped_from_odom.pose.position.y)
                    pass

                # 目指すべき場所
                dest_pos_from_odom = PoseStamped()
                dest_pos_from_odom.header = self.nearest_pose_stamped_from_odom.header
                dest_pos_from_odom.pose.position.x = self.kalman_filter.state[0]
                dest_pos_from_odom.pose.position.y = self.kalman_filter.state[1]



                vel_from_odom = PoseStamped()
                vel_from_odom.header = self.nearest_pose_stamped_from_odom.header
                vel_from_odom.pose.position.x = (dest_pos_from_odom.pose.position.x - self.pre_dest_pos_from_odom.pose.position.x)/self.delta_t + transform_to_odom.transform.translation.x
                vel_from_odom.pose.position.y = (dest_pos_from_odom.pose.position.y - self.pre_dest_pos_from_odom.pose.position.y)/self.delta_t + transform_to_odom.transform.translation.y

                self.pre_dest_pos_from_odom = dest_pos_from_odom

                # odom視点のものをspot(body)視点にする
                self.dest_pos_from_spot = tf2_geometry_msgs.do_transform_pose(dest_pos_from_odom, transform_to_body)
                vel_from_spot = tf2_geometry_msgs.do_transform_pose(vel_from_odom, transform_to_body)


                self.send_pose_array:PoseArray = PoseArray()
                pos = Pose()
                pos.position.x = self.dest_pos_from_spot.pose.position.x
                pos.position.y = self.dest_pos_from_spot.pose.position.y
                vel = Pose()
                vel.position.x = vel_from_spot.pose.position.x
                vel.position.y = vel_from_spot.pose.position.y

                self.send_pose_array.poses.append(pos)
                self.send_pose_array.poses.append(vel)

                #描画
                self.observed_circle.x = self.nearest_pose_stamped_from_spot.pose.position.x*self.viz_scale + offset_x
                self.observed_circle.y = self.nearest_pose_stamped_from_spot.pose.position.y*self.viz_scale + offset_y
                self.estimated_circle.x = self.dest_pos_from_spot.pose.position.x*self.viz_scale+offset_x
                self.estimated_circle.y = self.dest_pos_from_spot.pose.position.y*self.viz_scale+offset_y
                self.estimated_circle.radius = (min(self.kalman_filter.cov[0,0], self.kalman_filter.cov[1,1])**0.5)*self.viz_scale

                self.vel.x = self.dest_pos_from_spot.pose.position.x
                self.vel.y = self.dest_pos_from_spot.pose.position.y
                self.vel.x2 = self.dest_pos_from_spot.pose.position.x + vel_from_spot.pose.position.x
                self.vel.y2 = self.dest_pos_from_spot.pose.position.y + vel_from_spot.pose.position.y
                print("==dest_pos==")
                print(self.dest_pos_from_spot.pose.position)


                self.pub.publish(self.dest_pos_from_spot)

                # 認識した人の位置をtfにpublishする
                position = self.nearest_pose_stamped_from_spot.pose.position
                orientation = self.nearest_pose_stamped_from_spot.pose.orientation
                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = target_id
                t.child_frame_id = "nearest_person"
                t.transform.translation.x = position.x
                t.transform.translation.y = position.y
                t.transform.translation.z = position.z
                t.transform.rotation.x = orientation.x
                t.transform.rotation.y = orientation.y
                t.transform.rotation.z = orientation.z
                t.transform.rotation.w = orientation.w
                self.broadcaster.sendTransform(t)

                # 指定した周期で回るはず
                self.rate.sleep()


if __name__ == "__main__":
    RelativeTfConverter()
    print("waiting for viz_thread termination...")
    viz_thread.join()

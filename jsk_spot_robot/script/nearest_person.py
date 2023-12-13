import rospy
import tf2_ros
import tf2_geometry_msgs
from jsk_recognition_msgs.msg import BoundingBoxArray
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
import sys


        
class RelativeTfConverter:
    def __init__(self):
        rospy.init_node("rel_tf_node")

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(100))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.broadcaster = tf2_ros.TransformBroadcaster()
        rospy.Subscriber("/spot_recognition/bbox_array", BoundingBoxArray, self.callback)

        self.pub = rospy.Publisher("/nearest_person", PoseStamped)
        rospy.spin()

    def cvtBox2PoseStamped(self, box):
        original_pose_stamped = PoseStamped()
        original_pose_stamped.header = box.header
        original_pose_stamped.pose = box.pose
        return original_pose_stamped

    
    def callback(self, msg:BoundingBoxArray):
        try:
            from_id = "odom"
            target_id = "body"
            transform = self.tf_buffer.lookup_transform(target_id, from_id, rospy.Time(0), rospy.Duration(1.0))
            
            if len(msg.boxes) > 0:
                self.nearest_pose_stamped = PoseStamped()
                self.nearest_distance = 1e10
                self.updated = False
                for i, box in enumerate(msg.boxes):
                    pose_stamped = self.cvtBox2PoseStamped(box)
                    pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
                    # print(f"detected_{i}: {pose_transformed}")
                    distance = (pose_transformed.pose.position.x**2 + pose_transformed.pose.position.y**2+ pose_transformed.pose.position.z**2)**0.5
                    # print(f"id={i}, pos={pose_transformed.pose.position}")
                    if (distance < self.nearest_distance and pose_transformed.pose.position.x < 0 and abs(pose_transformed.pose.position.y) < 2):
                        self.nearest_distance = distance
                        self.nearest_pose_stamped= pose_transformed
                        self.updated = True
                
                if self.updated:
                    self.pub.publish(self.nearest_pose_stamped)
                    # print(f"publish {self.nearest_pose_stamped.pose.position}")
                    position = self.nearest_pose_stamped.pose.position
                    orientation = self.nearest_pose_stamped.pose.orientation
                    t = TransformStamped()
                    t.header.stamp = rospy.Time.now()
                    t.header.frame_id = target_id
                    t.child_frame_id = "nearest_person"
                    print(position)
                    t.transform.translation.x = position.x
                    t.transform.translation.y = position.y
                    t.transform.translation.z = position.z
                    t.transform.rotation.x = orientation.x
                    t.transform.rotation.y = orientation.y
                    t.transform.rotation.z = orientation.z
                    t.transform.rotation.w = orientation.w
                    self.broadcaster.sendTransform(t)
        except:
            pass


if __name__ == "__main__":
    RelativeTfConverter()

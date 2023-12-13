#!/usr/bin/env roseus

(load "start-spot.l")

(ros::roseus-add-msgs "jsk_recognition_msgs")
(ros::roseus "centering" :anonymous t)
(defun pose-cb (msg)
	(if (> (list-length (send msg :poses)) 0) 
		(progn
			(setq found_people_num (list-length (send msg :poses)))
			(setq found_joints_num (list-length (send (elt (send msg :poses) 0) :poses)))
			(setq x 0)
			(setq count 0)
			;; (print (send (elt (send (elt (send msg :poses) 0) :poses) 0) :position :x))
			(dotimes (i found_joints_num)
				(setq x (+ x (send (elt (send (elt (send msg :poses) 0) :poses) i) :position :x)))
				(setq count (+ count 1))
			)
			(setq x (/ x count))
			(format t "found ~A people " found_people_num)
			(format t "x = ~A ~%" x)
			(setq Kp  (/ 3.14 12))
			(setq diff (- x 320))
			(cond 
				((or (> diff 30) (< diff -30))
							(send *ri* :go-velocity 0 0 (* (/ diff -320) Kp) 500)
				)
			)
		)
		(progn
			(print 'not_found)
		)
	)
)

(print 'hoge)
(ros::subscribe "/edgetpu_human_pose_estimator/output/poses" jsk_recognition_msgs::PeoplePoseArray #'pose-cb)
(ros::spin)
(exit)

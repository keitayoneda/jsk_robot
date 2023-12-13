#!/usr/bin/env roseus

(load "start-spot.l")

(ros::roseus-add-msgs "geometry_msgs")

(ros::roseus "centering" :anonymous t)

(defclass lpf
	:super propertied-object
	:slots (m_coeff m_val)
)

(defmethod lpf
	(:init (coeff)
		(setq m_coeff coeff)
		(setq m_val 0)
	)
)

(defmethod lpf
	(:filter (value)
		(setq m_val (+ (* (- 1 m_coeff) value) (* m_coeff m_val)))
	)
)

(defmethod lpf
	(:get nil
		m_val
	)
)

(setq x_filtered (instance lpf :init 0.8))
(setq y_filtered (instance lpf :init 0.8))

(defun pose-cb (msg)
	(setq dest_x 0)
	(setq dest_y 0)
	(setq x (send msg :pose :position :x))
	(setq y (send msg :pose :position :y))
	(send x_filtered :filter x)
	(send y_filtered :filter y)
	(setq delta_x (- (send x_filtered :get) dest_x))
	(setq delta_y (- (send y_filtered :get) dest_y))
	(setq move_distance (sqrt (+ (* delta_x delta_x) (* delta_y delta_y))))
	(setq alpha (/ (- move_distance 1) move_distance))
	(setq move_x (* alpha delta_x))
	(setq move_y (* alpha delta_y))
	(setq move_th (rad2deg (atan2 (* delta_y -1) (* delta_x -1))))
	(setq move_th (* move_th 0.8))
	(setq delta_t_us 500000) ;;0.5秒おきに司令を送る

	(format t "move_distance = ~A ~%" move_distance)
	(format t "found person = (~A, ~A) ~%" x y)
	(format t "go-pos ~A ~A ~A ~%" move_x move_y move_th)
	(if (< move_distance 2) 
		(progn
			(send *ri* :go-pos move_x move_y move_th :wait nil)
			(unix:usleep delta_t_us) 
		)
		(progn
			(format t "target pos is too far ~%")
		)
	)
)

(setq *detected_raw* (make-cube 5 5 5))
(setq *detected_filtered* (make-cube 5 5 5))
(objects (list *spot*))

(print 'start_program)
(ros::subscribe "/nearest_person" geometry_msgs::PoseStamped #'pose-cb)
(ros::spin)
(exit)

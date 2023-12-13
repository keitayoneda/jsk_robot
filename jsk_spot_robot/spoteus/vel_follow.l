#!/usr/bin/env roseus

(load "start-spot.l")

(ros::roseus-add-msgs "geometry_msgs")

(ros::roseus "centering" :anonymous t)

;; LPF
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

;; PIレギュレータ
(defclass PI_regulator
 :super propertied-object
 :slots (m_p_gain m_i_gain m_integrated)
)
(defmethod PI_regulator
  (:init (p_gain i_gain)
	  (setq m_p_gain p_gain)
		(setq m_i_gain i_gain)
		(setq m_integrated 0)
	)
)
(defmethod PI_regulator
  (:reset nil
    (setq m_integrated 0)
	)
)
(defmethod PI_regulator
	(:update (delta)
		(setq m_integrated (+ m_integrated delta))

	)
)
(defmethod PI_regulator
  (:get (delta)
		(let (ret)
			(setq m_integrated (+ m_integrated delta))
			(setq ret (+ (* m_p_gain delta) (* m_i_gain m_integrated)))
			ret
		)
	)
)


(setq x_filtered (instance lpf :init 0.8))
(setq y_filtered (instance lpf :init 0.8))
(setq x_p_gain 1.0)
(setq x_i_gain 0.0)
(setq y_p_gain 1.0)
(setq y_i_gain 0.0)
(setq x_pi_regulator (instance PI_regulator :init x_p_gain x_i_gain))
(setq y_pi_regulator (instance PI_regulator :init y_p_gain y_i_gain))

(defun pose-cb (msg)
	(setq dest_x -1)
	(setq dest_y 0)
	(setq x (send msg :pose :position :x))
	(setq y (send msg :pose :position :y))
	(send x_filtered :filter x)
	(send y_filtered :filter y)
	(setq delta_x (- (send x_filtered :get) dest_x)) ;;delta_x = x_filtered - dest_x
	(setq delta_y (- (send y_filtered :get) dest_y)) ;;delta_y = y_filtered - dest_y
	(format t "delta_x ~A ~%" delta_x)
	(setq vx (send x_pi_regulator :get delta_x)) ;; x_pi_regulatorに突っ込む
	(setq vy (send y_pi_regulator :get delta_y)) ;; y_pi_regulatorに突っ込む
	(setq abs_vel (sqrt (+ (* vx vx) (* vy vy)))) ;; 
	;;(setq alpha (/ (- move_distance 1) move_distance)) ;; 認識した人に対して1m距離をとった位置をキープする
	;;(setq move_x (* alpha delta_x))
	;;(setq move_y (* alpha delta_y))
	(setq vth (atan2 (* delta_y -1) (* delta_x -1)))

	;;(format t "move_distance = ~A ~%" move_distance)
	(format t "found person = (~A, ~A) ~%" x y)
	(if (< abs_vel 3) 
		(progn
			(format t "go-velocity ~A ~A ~A ~%" vx vy vth)
			(send *ri* :go-velocity vx vy vth )
			(unix:usleep 500000)
		)
		(progn
			(format t "target vel is too large ~%")
		)
	)
)


(print 'start)
(ros::subscribe "/nearest_person" geometry_msgs::PoseStamped #'pose-cb)
(ros::spin)
(exit)

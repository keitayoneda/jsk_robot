#!/usr/bin/env roseus

(load "start-spot.l")

(ros::roseus-add-msgs "geometry_msgs")

(ros::roseus "vel_control" :anonymous t)

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

;; PIレギュレータ
(defclass BufferdPIRegulator
 :super propertied-object
 :slots (m_p_gain m_i_gain m_integrated m_buf m_buf_num)
)
(defmethod BufferdPIRegulator
  (:init (p_gain i_gain buf_num)
	  (setq m_p_gain p_gain)
		(setq m_i_gain i_gain)
		(setq m_integrated 0)
		(setq m_buf (make-list buf_num :initial-element 0))
		(setq m_buf_num buf_num)
	)
)
(defmethod BufferdPIRegulator
  (:reset nil
    (setq m_integrated 0)
		(setq m_buf (make-list buf_num :initial-element 0))
	)
)
(defmethod BufferdPIRegulator
	(:update (delta)
		(setq m_integrated (+ m_integrated delta))
		(setq m_integrated (- m_integrated (car (last m_buf))))
		(push delta m_buf)
		(setq m_buf (butlast m_buf))
	)
)
(defmethod BufferdPIRegulator
  (:get (delta)
		(let (ret)
			(setq m_integrated (+ m_integrated delta))
			(setq m_integrated (- m_integrated (car (last m_buf))))
			(push delta m_buf)
			(setq m_buf (butlast m_buf))
			(setq ret (+ (* m_p_gain delta) (* m_i_gain m_integrated)))
			ret
		)
	)
)


(setq th_filtered (instance lpf :init 0.1))

(defun pose-cb (msg)
	(setq pos (elt (send msg :poses) 0))
	(setq vel (elt (send msg :poses) 1))
	(print pos)
	(print vel)
	(setq move_x (* 0.6 (send  pos :position :x)))
	(setq move_y (* 0.6 (send  pos :position :y)))
	(setq th (* 1.0 (atan2 (* move_y -1) (* move_x -1))))
	(send th_filtered :filter th)
	(setq move_th (send th_filtered :get))
	(setq abs_vel (sqrt (+ (* move_x move_x) (* move_y move_y)))) ;; 
	(setq sign (/ move_th (abs move_th)))
	(setq move_th (* sign (min (abs move_th) (/ (abs move_th) (* abs_vel 1.5)))))

	(if (< abs_vel 3) 
		(progn
			(format t "go-velocity ~A ~A ~A ~%" move_x move_y move_th)
			(send *ri* :go-velocity move_x move_y move_th 700)
			(unix:usleep 300000)
		)
		(progn
			(format t "target vel is too large! vel= ~A ~%" abs_vel)
			(format t "vel= ~A ~A ~A ~%" move_x move_y move_th)
		)
	)
)


(print 'start)
(ros::subscribe "/dest_pos_vel" geometry_msgs::PoseArray #'pose-cb)
(ros::spin)
(exit)

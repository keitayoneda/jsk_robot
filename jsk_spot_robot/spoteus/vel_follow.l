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


(setq x_filtered (instance lpf :init 0.8))
(setq y_filtered (instance lpf :init 0.8))
(setq th_filtered (instance lpf :init 0.1))
(setq x_p_gain 1.0)
(setq x_i_gain 0.01)
(setq y_p_gain 1.0)
(setq y_i_gain 0.01)
(setq x_pi_regulator (instance BufferdPIregulator :init x_p_gain x_i_gain 3))
(setq y_pi_regulator (instance BufferdPIregulator :init y_p_gain y_i_gain 3))

(defun pose-cb (msg)
	;;(setq dest_x 0)
	;;(setq dest_y 0)
	;;(setq x (send msg :pose :position :x))
	;;(setq y (send msg :pose :position :y))
	;;(send x_filtered :filter x)
	;;(send y_filtered :filter y)
	;;(setq delta_x (- (send x_filtered :get) dest_x)) ;;delta_x = x_filtered - dest_x
	;;(setq delta_y (- (send y_filtered :get) dest_y)) ;;delta_y = y_filtered - dest_y
	;;(format t "delta_x ~A ~%" delta_x)
	;;(setq move_distance (sqrt (+ (* delta_x delta_x) (* delta_y delta_y))))
	;;(setq vx (send x_pi_regulator :get delta_x)) ;; x_pi_regulatorに突っ込む
	;;(setq vy (send y_pi_regulator :get delta_y)) ;; y_pi_regulatorに突っ込む
	;;(setq alpha (/ (- move_distance 1) move_distance)) ;; 認識した人に対して1m距離をとった位置をキープする
	;;(setq move_x (* alpha vx))
	;;(setq move_y (* alpha vy))
	(setq move_x (* 0.6 (send  msg :pose :position :x)))
	(setq move_y (* 0.6 (send  msg :pose :position :y)))
	(setq th (* 1.0 (atan2 (* move_y -1) (* move_x -1))))
	(send th_filtered :filter th)
	(setq move_th (send th_filtered :get))
	(setq abs_vel (sqrt (+ (* move_x move_x) (* move_y move_y)))) ;; 
	(setq sign (/ move_th (abs move_th)))
	(setq move_th (* sign (min (abs move_th) (/ (abs move_th) (* abs_vel 3)))))

	;; (format t "found person = (~A, ~A) ~%" x y)
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
(ros::subscribe "/nearest_person" geometry_msgs::PoseStamped #'pose-cb)
(ros::spin)
(exit)

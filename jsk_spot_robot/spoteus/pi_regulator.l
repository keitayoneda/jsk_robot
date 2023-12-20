#!/usr/bin/env roseus



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
		;;(let (ret)
			;;(setq m_integrated (+ m_integrated delta))
			;;(setq ret (+ (* m_p_gain delta) (* m_i_gain m_integrated)))
			;;ret
		;;)
	)
)

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

(setq x_p_gain -1.0)
(setq x_i_gain -0.0)
(setq y_p_gain -1.0)
(setq y_i_gain -0.0)
(setq x_pi_regulator (instance PI_regulator :init x_p_gain x_i_gain))
(setq y_pi_regulator (instance BufferdPIRegulator :init y_p_gain y_i_gain 3))

(setq output (send x_pi_regulator :get 0))
(setq output (send y_pi_regulator :update 1))
(setq output (send y_pi_regulator :update 2))
(setq output (send y_pi_regulator :update 3))
(setq output (send y_pi_regulator :update 4))

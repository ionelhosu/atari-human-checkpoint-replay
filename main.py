from database import *
from emulator import *
import tensorflow as tf
import numpy as np
import time
from ale_python_interface import ALEInterface
import cv2
from scipy import misc
import gc #garbage colloector
import thread

gc.enable()

params = {
	'visualize' : True,
	'network_type':'nips',
	'ckpt_file':None,
	'steps_per_epoch': 50000,
	'num_epochs': 250,
	'eval_freq':50000,
	'steps_per_eval':10000,
	'copy_freq' : 10000,
	'disp_freq':10000,
	'save_interval':10000,
	'db_size': 1000000,
	'batch': 32,
	'num_act': 0,
	'input_dims' : [210, 160, 3],
	'input_dims_proc' : [84, 84, 4],
	'learning_interval': 1,
	'eps': 0.1,
	'eps_step':1000000,
	'eps_min' : 0.1,
	'eps_eval' : 0.00,
	'discount': 0.95,
	'lr': 0.0002,
	'rms_decay':0.99,
	'rms_eps':1e-6,
	'train_start':100,
	'img_scale':255.0,
	'clip_delta' : 0, #nature : 1
	'gpu_fraction' : 0.9,
	'batch_accumulator':'mean',
	#'num_threads' : 4,
	'record_eval' : True,
	'only_eval' : 'n',
	'frames_per_episode' : 1800
}

class deep_atari:
	def __init__(self,params):
		print 'Initializing Module...'
		self.params = params

		self.gpu_config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.params['gpu_fraction']))

		self.sess = tf.Session(config=self.gpu_config)
		self.DB = database(self.params)
		self.engine = emulator(rom_name='private_eye.bin', vis=self.params['visualize'],windowname=self.params['network_type']+'_preview')
		self.params['num_act'] = len(self.engine.legal_actions)
		self.build_net()
		self.training = True
		self.lock = thread.allocate_lock()	

	def build_net(self):
		print 'Building QNet and targetnet...'		
		self.qnet = DQN(self.params,'qnet')
		self.targetnet = DQN(self.params,'targetnet')
		self.sess.run(tf.initialize_all_variables())
		saver_dict = {'qw1':self.qnet.w1,'qb1':self.qnet.b1,
				'qw2':self.qnet.w2,'qb2':self.qnet.b2,
				'qw3':self.qnet.w3,'qb3':self.qnet.b3,
				'qw4':self.qnet.w4,'qb4':self.qnet.b4,
				'qw5':self.qnet.w5,'qb5':self.qnet.b5,
				'tw1':self.targetnet.w1,'tb1':self.targetnet.b1,
				'tw2':self.targetnet.w2,'tb2':self.targetnet.b2,
				'tw3':self.targetnet.w3,'tb3':self.targetnet.b3,
				'tw4':self.targetnet.w4,'tb4':self.targetnet.b4,
				'tw5':self.targetnet.w5,'tb5':self.targetnet.b5,
				'step':self.qnet.global_step}
		self.saver = tf.train.Saver(saver_dict)
		#self.saver = tf.train.Saver()
		self.cp_ops = [
			self.targetnet.w1.assign(self.qnet.w1),self.targetnet.b1.assign(self.qnet.b1),
			self.targetnet.w2.assign(self.qnet.w2),self.targetnet.b2.assign(self.qnet.b2),
			self.targetnet.w3.assign(self.qnet.w3),self.targetnet.b3.assign(self.qnet.b3),
			self.targetnet.w4.assign(self.qnet.w4),self.targetnet.b4.assign(self.qnet.b4),
			self.targetnet.w5.assign(self.qnet.w5),self.targetnet.b5.assign(self.qnet.b5)]
		
		self.sess.run(self.cp_ops)
		
		if self.params['ckpt_file'] is not None:
			print 'loading checkpoint : ' + self.params['ckpt_file']
			self.saver.restore(self.sess,self.params['ckpt_file'])
			temp_train_cnt = self.sess.run(self.qnet.global_step)
			temp_step = temp_train_cnt * self.params['learning_interval']
			print 'Continue from'
			print '        -> Steps : ' + str(temp_step)
			print '        -> Minibatch update : ' + str(temp_train_cnt)
			
	def do_training(self,th_idx):
		#print 'Training thread ' + str(th_idx) + ' initiated'
		print 'Training thread initiated'
		while True:
			if self.training and self.step % self.params['learning_interval'] == 0 and self.DB.get_size() > self.params['train_start'] :
				bat_s,bat_a,bat_t,bat_n,bat_r = self.DB.get_batches()
				bat_a = self.get_onehot(bat_a)	
				
				if self.params['copy_freq'] > 0 :
					feed_dict={self.targetnet.x: bat_n}
					self.lock.acquire()
					q_t = self.sess.run(self.targetnet.y,feed_dict=feed_dict)
					self.lock.release()
				else:
					feed_dict={self.qnet.x: bat_n}
					self.lock.acquire()
					q_t = self.sess.run(self.qnet.y,feed_dict=feed_dict)
					self.lock.release()		
				q_t = np.amax(q_t,axis=1)
				#print str(th_idx) + '_qt_old : '
				#print q_t
								
				feed_dict={self.qnet.x: bat_s, self.qnet.q_t: q_t, self.qnet.actions: bat_a, self.qnet.terminals:bat_t, self.qnet.rewards: bat_r}
				#print str(th_idx) + '_old : '
				#print self.sess.run(self.qnet.b4)
				self.lock.acquire()
				_,self.train_cnt,self.cost = self.sess.run([self.qnet.rmsprop,self.qnet.global_step,self.qnet.cost],feed_dict=feed_dict)
				#print str(th_idx) + '_new : '
				#print self.sess.run(self.qnet.b4)
				self.lock.release()		
				self.total_cost_train += np.sqrt(self.cost)
				self.train_cnt_for_disp += 1
	


	def start(self):
		self.reset_game()
		self.step = 0
		self.reset_statistics('all')
		self.train_cnt = self.sess.run(self.qnet.global_step)

		if self.train_cnt > 0 :
			self.step = self.train_cnt * self.params['learning_interval']
			try:
				self.log_train = open('log_training_'+self.params['network_type']+'.csv','a')
			except:
				self.log_train = open('log_training_'+self.params['network_type']+'.csv','w')
				self.log_train.write('step,epoch,train_cnt,avg_reward,avg_q,epsilon,time\n')	

			try:
				self.log_eval = open('log_eval_'+self.params['network_type']+'.csv','a')
			except:
				self.log_eval = open('log_eval_'+self.params['network_type']+'.csv','w')
				self.log_eval.write('step,epoch,train_cnt,avg_reward,avg_q,epsilon,time\n')
		else:
				self.log_train = open('log_training_'+self.params['network_type']+'.csv','w')
				self.log_train.write('step,epoch,train_cnt,avg_reward,avg_q,epsilon,time\n')	
				self.log_eval = open('log_eval_'+self.params['network_type']+'.csv','w')
				self.log_eval.write('step,epoch,train_cnt,avg_reward,avg_q,epsilon,time\n')

		#for ii in range(self.params['num_threads']):
		thread.start_new_thread(self.do_training,(0,))			
		time.sleep(1.5)
		self.s = time.time()
		print self.params
		print 'Start training!'
		print 'Collecting replay memory for ' + str(self.params['train_start']) + ' steps'

		while self.step < (self.params['steps_per_epoch'] * self.params['num_epochs'] * self.params['learning_interval'] + self.params['train_start']):

			#if self.engine.get_frame_number() >= self.params['frames_per_episode']:
			#	self.reset_game()
					
			if self.training : 
				if self.DB.get_size() >= self.params['train_start'] : self.step += 1 ; self.steps_train += 1
			else : self.step_eval += 1
			if self.state_gray_old is not None and self.training:
				self.DB.insert(self.state_gray_old[26:110,:],self.reward_scaled,self.action_idx,self.terminal)

			if self.training and self.params['copy_freq'] > 0 and self.step % self.params['copy_freq'] == 0 and self.DB.get_size() > self.params['train_start']:
				print '&&& Copying Qnet to targetnet\n'
				self.lock.acquire()
				self.sess.run(self.cp_ops)
				self.lock.release()

			if self.training :				
				self.params['eps'] = max(self.params['eps_min'],1.0 - float(self.step)/float(self.params['eps_step']))
			else:
				self.params['eps'] = 0.05

			if self.DB.get_size() > self.params['train_start'] and self.step % self.params['save_interval'] == 0 and self.training:
				save_idx = self.train_cnt
				self.lock.acquire()
				self.saver.save(self.sess,'ckpt/model_'+self.params['network_type']+'_'+str(save_idx))
				self.lock.release()
				sys.stdout.write('$$$ Model saved : %s\n\n' % ('ckpt/model_'+self.params['network_type']+'_'+str(save_idx)))
				sys.stdout.flush()

			if self.training and self.step > 0 and self.step % self.params['disp_freq']  == 0 and self.DB.get_size() > self.params['train_start'] : 
				self.write_log_train()

			if self.training and self.step > 0 and self.step % self.params['eval_freq'] == 0 and self.DB.get_size() > self.params['train_start'] : 
				
				self.reset_game()
				if self.step % self.params['steps_per_epoch'] == 0 : self.reset_statistics('all')
				else: self.reset_statistics('eval')
				self.training = False
				#TODO : add video recording				
				continue
			if self.training and self.step > 0 and self.step % self.params['steps_per_epoch'] == 0 and self.DB.get_size() > self.params['train_start']: 
				self.reset_game()
				self.reset_statistics('all')
				#self.training = False
				continue

			if not self.training and self.step_eval >= self.params['steps_per_eval'] :
				self.write_log_eval()
				self.reset_game()
				self.reset_statistics('eval')
				self.training = True
				continue
			

			if self.terminal or (self.engine.get_frame_number() >= self.params['frames_per_episode']):  
				self.reset_game()
				if self.training : 
					self.num_epi_train += 1 
					self.total_reward_train += self.epi_reward_train
					self.epi_reward_train = 0
				else : 
					self.num_epi_eval += 1 
					self.total_reward_eval += self.epi_reward_eval
					self.epi_reward_eval = 0
				continue

			self.action_idx,self.action, self.maxQ = self.select_action(self.state_proc)
			self.state, self.reward, self.terminal = self.engine.next(self.action)
			self.reward_scaled = self.reward // max(1,abs(self.reward))
			if self.training : self.epi_reward_train += self.reward ; self.total_Q_train += self.maxQ
			else : self.epi_reward_eval += self.reward ; self.total_Q_eval += self.maxQ	

			self.state_gray_old = np.copy(self.state_gray)
			self.state_proc[:,:,0:3] = self.state_proc[:,:,1:4]
			self.state_resized = cv2.resize(self.state,(84,110))
			self.state_gray = cv2.cvtColor(self.state_resized, cv2.COLOR_BGR2GRAY)
			self.state_proc[:,:,3] = self.state_gray[26:110,:]/self.params['img_scale']
			
			#TODO : add video recording
			if self.params['only_eval'] == 'y':
				time.sleep(0.01)

	def reset_game(self):
		self.state_proc = np.zeros((84,84,4)); self.action = -1; self.terminal = False; self.reward = 0
		self.state = self.engine.newGame()		
		self.state_resized = cv2.resize(self.state,(84,110))
		self.state_gray = cv2.cvtColor(self.state_resized, cv2.COLOR_BGR2GRAY)
		self.state_gray_old = None
		self.state_proc[:,:,3] = self.state_gray[26:110,:]/self.params['img_scale']

	def reset_statistics(self,mode):
		if mode == 'all':
			self.epi_reward_train = 0
			self.epi_Q_train = 0
			self.num_epi_train = 0
			self.total_reward_train = 0
			self.total_Q_train = 0
			self.total_cost_train = 0
			self.steps_train = 0
			self.train_cnt_for_disp = 0
		self.step_eval = 0
		self.epi_reward_eval = 0
		self.epi_Q_eval = 0		
		self.num_epi_eval = 0		
		self.total_reward_eval = 0
		self.total_Q_eval = 0


	def write_log_train(self):
		sys.stdout.write('### Training (Step : %d , Minibatch update : %d , Epoch %d)\n' % (self.step,self.train_cnt,self.step//self.params['steps_per_epoch'] ))

		sys.stdout.write('    Num.Episodes : %d , Avg.reward : %.3f , Avg.Q : %.3f, Avg.loss : %.3f\n' % (self.num_epi_train,float(self.total_reward_train)/max(1,self.num_epi_train),float(self.total_Q_train)/max(1,self.steps_train),self.total_cost_train/max(1,self.train_cnt_for_disp)))
		sys.stdout.write('    Epsilon : %.3f , Elapsed time : %.1f\n\n' % (self.params['eps'],time.time()-self.s))
		sys.stdout.flush()
		self.log_train.write(str(self.step) + ',' + str(self.step//self.params['steps_per_epoch']) + ',' + str(self.train_cnt) + ',')
		self.log_train.write(str(float(self.total_reward_train)/max(1,self.num_epi_train)) +','+ str(float(self.total_Q_train)/max(1,self.steps_train)) +',')
		self.log_train.write(str(self.params['eps']) +','+ str(time.time()-self.s) + '\n')
		self.log_train.flush()		
	
	def write_log_eval(self):
		sys.stdout.write('@@@ Evaluation (Step : %d , Minibatch update : %d , Epoch %d)\n' % (self.step,self.train_cnt,self.step//self.params['steps_per_epoch'] ))
		sys.stdout.write('    Num.Episodes : %d , Avg.reward : %.3f , Avg.Q : %.3f\n' % (self.num_epi_eval,float(self.total_reward_eval)/max(1,self.num_epi_eval),float(self.total_Q_eval)/max(1,self.params['steps_per_eval'])))
		sys.stdout.write('    Epsilon : %.3f , Elapsed time : %.1f\n\n' % (self.params['eps'],time.time()-self.s))
		sys.stdout.flush()
		self.log_eval.write(str(self.step) + ',' + str(self.step//self.params['steps_per_epoch']) + ',' + str(self.train_cnt) + ',')
		self.log_eval.write(str(float(self.total_reward_eval)/max(1,self.num_epi_eval)) +','+ str(float(self.total_Q_eval)/max(1,self.params['steps_per_eval'])) +',')
		self.log_eval.write(str(self.params['eps']) +','+ str(time.time()-self.s) + '\n')
		self.log_eval.flush()

	def select_action(self,st):
		if np.random.rand() > self.params['eps']:
			#greedy with random tie-breaking
			self.lock.acquire()
			Q_pred = self.sess.run(self.qnet.y, feed_dict = {self.qnet.x: np.reshape(st, (1,84,84,4))})[0] 
			self.lock.release()
			a_winner = np.argwhere(Q_pred == np.amax(Q_pred))
			if len(a_winner) > 1:
				act_idx = a_winner[np.random.randint(0, len(a_winner))][0]
				return act_idx,self.engine.legal_actions[act_idx], np.amax(Q_pred)
			else:
				act_idx = a_winner[0][0]
				return act_idx,self.engine.legal_actions[act_idx], np.amax(Q_pred)
		else:
			#random
			act_idx = np.random.randint(0,len(self.engine.legal_actions))
			self.lock.acquire()
			Q_pred = self.sess.run(self.qnet.y, feed_dict = {self.qnet.x: np.reshape(st, (1,84,84,4))})[0]
			self.lock.release()
			return act_idx,self.engine.legal_actions[act_idx], Q_pred[act_idx]

	def get_onehot(self,actions):
		actions_onehot = np.zeros((self.params['batch'], self.params['num_act']))
		
		for i in range(self.params['batch']):
			actions_onehot[i,actions[i]] = 1
		return actions_onehot


if __name__ == "__main__":
	dict_items = params.items()
	for i in range(1,len(sys.argv),2):
		if sys.argv[i] == '-weight' :params['ckpt_file'] = sys.argv[i+1]
		elif sys.argv[i] == '-network_type' :params['network_type'] = sys.argv[i+1]
		elif sys.argv[i] == '-visualize' :
			if sys.argv[i+1] == 'y' : params['visualize'] = True
			elif sys.argv[i+1] == 'n' : params['visualize'] = False
			else:
				print 'Invalid visualization argument!!! Available arguments are'
				print '        y or n'
				raise ValueError()
		elif sys.argv[i] == '-gpu_fraction' : params['gpu_fraction'] = float(sys.argv[i+1])
		elif sys.argv[i] == '-db_size' : params['db_size'] = int(sys.argv[i+1])
		#elif sys.argv[i] == '-num_threads' : params['num_threads'] = int(sys.argv[i+1])
		elif sys.argv[i] == '-only_eval' : params['only_eval'] = sys.argv[i+1]
		else : 
			print 'Invalid arguments!!! Available arguments are'
			print '        -weight (filename)'
			print '        -network_type (nips or nature)'
			print '        -visualize (y or n)'
			print '        -gpu_fraction (0.1~0.9)'
			print '        -db_size (integer)'
			#print '        -num_threads (integer)'
			raise ValueError()
	if params['network_type'] == 'nips':
		from DQN_nips import *
	elif params['network_type'] == 'nature':
		from DQN_nature import *
		params['steps_per_epoch']= 200000
		params['eval_freq'] = 200000
		params['steps_per_eval'] = 10000
		params['copy_freq'] = 40000
		params['disp_freq'] = 20000
		params['save_interval'] = 20000
		params['learning_interval'] = 1
		params['discount'] = 0.99
		params['lr'] = 0.00025
		params['rms_decay'] = 0.95
		params['rms_eps']=0.01
		params['clip_delta'] = 1.0
		params['train_start']=10000
		params['batch_accumulator'] = 'sum'
		params['eps_step'] = 4000000
		params['num_epochs'] = 1000
		params['batch'] = 32
	else :
		print 'Invalid network type! Available network types are'
		print '        nips or nature'
		raise ValueError()

	if params['only_eval'] == 'y' : only_eval = True
	elif params['only_eval'] == 'n' : only_eval = False
	else :
		print 'Invalid only_eval option! Available options are'
		print '        y or n'
		raise ValueError()

	if only_eval:
		params['eval_freq'] = 1
		params['train_start'] = 100

	da = deep_atari(params)
	da.start()

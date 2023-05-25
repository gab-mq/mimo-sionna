import os

try:
    import sionna as sn
except:
    os.system("pip install sionna")
    import sionna as sn

from sionna.utils import BinarySource, ebnodb2no, QAMSource
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.channel import ApplyOFDMChannel, OFDMChannel, AWGN, gen_single_sector_topology as gen_topology, subcarrier_frequencies, GenerateOFDMChannel, ChannelModel
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.channel.tr38901 import Antenna, AntennaArray, CDL, UMi, UMa, SystemLevelChannel, SystemLevelScenario, UMaScenario, PanelArray
from sionna.mimo import StreamManagement
from sionna.channel.generate_ofdm_channel import cir_to_ofdm_channel
from sionna.utils.metrics import compute_ber
from sionna.channel.utils import generate_uts_topology, set_3gpp_scenario_parameters, relocate_uts, drop_uts_in_sector, sample_bernoulli, expand_to_rank

import numpy as np
from numpy.random import random, randint
import matplotlib.pyplot as plt
import tensorflow as tf
import math

class CellFree():

  def __init__(self,
               N=4, 
               L=4,
               K=None,
               K_L=3,
               N_BITS_PER_SYMBOL=2,
               N_BITS=1024,
               BATCH_SIZE=1,
               scenario='umi',
               CARRIER_FREQUENCY=3.5e9,
               EBNO_DB=20.0,
               CODERATE=0.5,
               STREAMS_PER_UE=2):
    '''
    N                   = Número de antenas por estação base
    L                   = Número de estações base
    K                   = Número de usuários
    K_L                 = Número de usuários por estação base
    N_BITS_PER_SYMBOL   = Número de bits por símbolo
    N_BITS              = Número total de bits
    BATCH_SIZE          = Quantidade de arranjos
    scenario            = Tipo de cenário MIMO
    CARRIER_FREQUENCY   = Frequência da portadora (Hz)
    EBNO_DB             = Relação de potência sinal-ruído (SNR) em decibéis
    CODERATE            = Taxa de código
    STREAMS_PER_UE      = Sequência de bits por usuário 
    '''

    self.N=N
    self.L=L
    if K==None:
      self.K=L*K_L
    else:
      self.K=K
    self.K_L=K_L
    self.N_BITS_PER_SYMBOL=N_BITS_PER_SYMBOL
    self.N_BITS=N_BITS
    self.BATCH_SIZE=BATCH_SIZE
    self.scenario=scenario
    self.CARRIER_FREQUENCY=CARRIER_FREQUENCY
    self.EBNO_DB=EBNO_DB
    self.CODERATE=CODERATE
    self.STREAMS_PER_UE=STREAMS_PER_UE
    self.ue_ap_association=self.gen_association()
  
  def gen_association(self):
    ue_ap_association=np.zeros([self.L,self.K])
    c,j=0,0
    for i in range(self.K):
      ue_ap_association[j,i]=1
      c+=1
      if c==self.K_L:
        c=0
        j+=1
    return ue_ap_association

  @property
  def channel_model(self):
    ue_array=Antenna(polarization='single',
                 polarization_type='V',
                 antenna_pattern='omni',
                 carrier_frequency=self.CARRIER_FREQUENCY)

    ap_array=AntennaArray(num_rows=self.N,
                          num_cols=1,
                          polarization='dual',
                          polarization_type='VH',
                          antenna_pattern="38.901",
                          carrier_frequency=self.CARRIER_FREQUENCY)

    if self.scenario=='umi':
      channel_model=UMi(carrier_frequency=self.CARRIER_FREQUENCY,
                        o2i_model='low',
                        ut_array=ue_array,
                        bs_array=ap_array,
                        direction='uplink',
                        enable_pathloss=False,
                        enable_shadow_fading=False)
    elif self.scenario=='uma':
      channel_model=UMa(carrier_frequency=self.CARRIER_FREQUENCY,
                        o2i_model='low',
                        ut_array=ue_array,
                        bs_array=ap_array,
                        direction='uplink',
                        enable_pathloss=False,
                        enable_shadow_fading=False)
    else:
      raise ValueError('scenario deve ser \'uma\' ou \'umi\'')

    topology=self.gen_my_topology(batch_size=self.BATCH_SIZE,
                                  num_ut=self.K,
                                  num_bs=self.L,
                                  scenario=self.scenario)
    channel_model.set_topology(*topology)
    return channel_model

  def show_topology(self):
    self.channel_model.show_topology()

  @property
  def stream_management(self):
    sm=StreamManagement(self.ue_ap_association,self.STREAMS_PER_UE)
    return sm

  @property
  def resource_grid(self):
    rg=ResourceGrid(num_ofdm_symbols=14,
                    fft_size=self.K*self.STREAMS_PER_UE*1,
                    subcarrier_spacing=30e3,
                    num_tx=self.K,
                    num_streams_per_tx=self.STREAMS_PER_UE,
                    cyclic_prefix_length=10,
                    pilot_pattern='kronecker',
                    pilot_ofdm_symbol_indices=[2,11])
    return rg

  @property
  def encoder(self):
    N_CODED_BITS=int(self.resource_grid.num_data_symbols*self.N_BITS_PER_SYMBOL)
    N_INFO_BITS=int(N_CODED_BITS*self.CODERATE)
    encoder=LDPC5GEncoder(N_INFO_BITS,N_CODED_BITS)
    return encoder
  
  @property
  def mapper(self):
    mapper=Mapper('qam',self.N_BITS_PER_SYMBOL)
    return mapper
  
  @property
  def rg_mapper(self):
    rg_mapper=ResourceGridMapper(self.resource_grid)
    return rg_mapper
  
  @property
  def ls_est(self):
    ls_est=LSChannelEstimator(self.resource_grid, interpolation_type='nn')
    return ls_est

  @property
  def lmmse_equ(self):    
    lmmse_equ=LMMSEEqualizer(self.resource_grid,self.stream_management)
    return lmmse_equ
  
  @property
  def demapper(self):
    demapper=Demapper('app','qam',self.N_BITS_PER_SYMBOL)
    return demapper
  
  @property
  def decoder(self):
    decoder=LDPC5GDecoder(self.encoder, hard_out=True)
    return decoder

  @property
  def channel_freq(self):
    channel_freq=ApplyOFDMChannel(add_awgn=True)
    return channel_freq
  
  @property
  def frequencies(self):
    frequencies=subcarrier_frequencies(self.resource_grid.fft_size, self.resource_grid.subcarrier_spacing)
    return frequencies

  def run(self):
    no=ebnodb2no(self.EBNO_DB,self.N_BITS_PER_SYMBOL,self.CODERATE,self.resource_grid)
    bs=BinarySource()
    b=bs([self.BATCH_SIZE, self.K, self.resource_grid.num_streams_per_tx, self.encoder.k])
    c=self.encoder(b)
    x=self.mapper(c)
    x_rg=self.rg_mapper(x)

    # a -> Coeficientes de caminho | tau -> Coeficientes de delay (s)
    a, tau=self.channel_model(
      num_time_samples=self.resource_grid.num_ofdm_symbols,
      sampling_frequency=1/self.resource_grid.ofdm_symbol_duration
    )
    h_freq=cir_to_ofdm_channel(self.frequencies, a, tau, normalize=True)

    y=self.channel_freq([x_rg,h_freq,no])
    h_hat, err_var=self.ls_est([y, no])
    x_hat, no_eff=self.lmmse_equ([y, h_hat, err_var, no])
    llr=self.demapper([x_hat, no_eff])
    b_hat=self.decoder(llr)
    print(f'BER: {compute_ber(b,b_hat)}')

  def gen_my_topology(self, batch_size,
                    num_ut,
                    num_bs,
                    scenario,
                    min_bs_ut_dist=None,
                    isd=None,
                    bs_height=None,
                    min_ut_height=None,
                    max_ut_height=None,
                    indoor_probability = None,
                    min_ut_velocity=None,
                    max_ut_velocity=None,
                    dtype=tf.complex64):
 

    PI=tf.constant(math.pi)

    params = set_3gpp_scenario_parameters(  scenario,
                                            min_bs_ut_dist,
                                            isd,
                                            bs_height,
                                            min_ut_height,
                                            max_ut_height,
                                            indoor_probability,
                                            min_ut_velocity,
                                            max_ut_velocity,
                                            dtype)
    min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height,\
            indoor_probability, min_ut_velocity, max_ut_velocity = params

    real_dtype = dtype.real_dtype

    if np.sqrt(num_bs)%1!=0:
      nbs=int(np.ceil(np.sqrt(num_bs)))**2
    else:
      nbs=num_bs

    # cria um vetor de range [0.5, SQRT(N_BS)] com intervalo unitário e normalizado por ISD (distancia entre duas APs)
    coords=tf.range(0.5,tf.math.sqrt(float(nbs)))*isd
    # cria uma grade de valores para determinar as coordenadas das APs
    X,Y=tf.meshgrid(coords,coords)
    X=tf.reshape(X,shape=-1)
    Y=tf.reshape(Y,shape=-1)
    X=tf.stack([X for i in range(batch_size)])
    Y=tf.stack([Y for i in range(batch_size)])

    bs_loc = tf.stack([X,
                       Y,
                      tf.fill( [batch_size, nbs], bs_height)], axis=-1)
    
    bs_xy_loc=tf.stack([
        X, Y
    ], axis=-1)

    # define a orientação das BSs apontando para o centro da seção
    sector_center = (min_bs_ut_dist + 0.5*isd)*0.5
    bs_downtilt = 0.5*PI - tf.math.atan(sector_center/bs_height)

    bs_orientation = tf.stack([ tf.zeros([batch_size, num_bs], real_dtype),
                                tf.fill([batch_size, num_bs], bs_downtilt),
                                tf.zeros([batch_size, num_bs], real_dtype)], axis=-1)

    ut_loc_list=[]
    ut_orientations_list=[]
    ut_velocities_list=[]
    in_state_list=[]
    n_ut_per_bs=num_ut//num_bs
    remaining=num_ut%num_bs
    for i in range(num_bs):
      ut_topology = generate_uts_topology(    batch_size,
                                          n_ut_per_bs+np.sign(remaining),
                                          'cell',
                                          bs_xy_loc[0,i],
                                          min_bs_ut_dist,
                                          isd,
                                          min_ut_height,
                                          max_ut_height,
                                          indoor_probability,
                                          min_ut_velocity,
                                          max_ut_velocity,
                                          dtype)
      
      ut_loc, ut_orientations, ut_velocities, in_state = ut_topology

      ut_loc_list.append(ut_loc)
      ut_orientations_list.append(ut_orientations)
      ut_velocities_list.append(ut_velocities)
      in_state_list.append(in_state)
    

      if remaining>0:
        remaining-=1

    ut_loc=tf.concat(ut_loc_list, 1)
    ut_orientations=tf.concat(ut_orientations_list, 1)
    ut_velocities=tf.concat(ut_velocities_list, 1)
    in_state=tf.concat(in_state_list,1)

    return ut_loc, bs_loc[:,0:num_bs], ut_orientations, bs_orientation[:,0:num_bs], ut_velocities,\
            in_state
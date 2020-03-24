import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_NAME = 'simulation'

flu_01 = json.loads(open('%s_with_flu.json' % EXPERIMENT_NAME).read())
flu_00 = json.loads(open('%s_without_flu.json' % EXPERIMENT_NAME).read())

PI_SUPPORT_01 = flu_01['range']
PI_SUPPORT_00 = flu_00['range']

post_flu_01 = flu_01['post']
post_flu_00 = flu_00['post']

plt.plot(PI_SUPPORT_01, post_flu_01,label='flu')
plt.plot(PI_SUPPORT_00, post_flu_00,label='no flu')
plt.legend()
plt.axis([0.0,1.0,0,1])
plt.xlabel('pi_c (proportion of population with COVID-19)')
plt.ylabel('Probability density')
plt.savefig('%s.png' % EXPERIMENT_NAME,dpi=150)

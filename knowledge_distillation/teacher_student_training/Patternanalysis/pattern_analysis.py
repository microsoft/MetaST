import pandas as pd
from collections import Counter
import operator

def processSlot(annotation):
	annotation = str(annotation).lower().strip(' .?!')
	result = []
	slots = {}
	for item in annotation.split('>'):
		item = str(item)
		if '</' in item:
			value = item.split('</')[0].strip()
			slot = item.split('</')[1].strip()
			if slot not in slots:
				slots[slot] = []
			slots[slot].append(value)
			tmp = '<{}>'.format(slot)
		elif '<' in item:
			tmp = item.split('<')[0].strip()
		else:
			tmp = item.strip()
		if len(tmp)>1:
			result.append(tmp)
	return ' '.join(result)

def processAllDetail(file):
	df = pd.read_csv(file, sep='\t', header=0, encoding = "utf-8")
	print (df.shape)
	# print (len(df['JudgedTaskunderstandingIntent']))
	df = df[(df['JudgedTaskunderstandingIntent'].isin(['send_text', 'send_text_meeting'])) | (df['CuIntents'].isin(['send_text', 'send_text_meeting']))]
	print (df.shape)
	# df2 = df[df['JudgedTaskunderstandingIntent'].isin(['send_text', 'send_text_meeting'])]
	# print (df2.shape)
	# df3 = df[df['CuIntents'].isin(['send_text', 'send_text_meeting'])]
	# print (df3.shape)
	# exit()
	processed_slots = df['JudgedTaskunderstandingMsgannotation'].map(processSlot)
	print (processed_slots)
	print (type(processed_slots))
	
	df['queryPattern'] = processed_slots
	print (df.shape)
	annotations = Counter(processed_slots)
	print ('Unique query patterns: {}'.format(len(annotations)))
	sorted_annotation = sorted(annotations.items(), key=operator.itemgetter(1), reverse=True)

	f1 = open(file.rstrip('.tsv')+'_QueryPatterns_all.tsv', 'w', encoding = 'utf-8')
	f1.writelines('Pattern\tFreq\tWrong\tAccuracy\n')
	ids_to_keep = []
	for item in sorted_annotation:
		# if item[1] == 1:
		#     break
		df_t = df[df['queryPattern']==item[0]]
		counts = df_t.AllCorrect.value_counts()
		# print (counts['T'])
		# exit()
		F=counts['F'] if 'F' in counts else 0
		T=counts['T'] if 'T' in counts else 0
		f1.writelines('{}\t{}\t{}\t{}\n'.format(item[0], item[1], F, T*1.0/item[1]))
		if F>0:
			df_t_F = df_t[df_t['AllCorrect']=='F']
			ids_to_keep +=list(df_t_F['ConversationId'])
		T_n = min(5, T)
		df_t_T = df_t[df_t['AllCorrect']=='T']
		if len(df_t_T)>0:
			df_t_T = df_t_T.sample(n=T_n) ##df_t_T might be 0 why?
		else:
			
			continue
		ids_to_keep +=list(df_t_T['ConversationId'])

	pat_1 = [item[0] for item in sorted_annotation if item[1]==1]
	df_t = df[df['queryPattern'].isin(pat_1)]
	ids_to_keep +=list(df_t['ConversationId'])

	print (len(ids_to_keep))
	exit()

	df_f = df[df['ConversationId'].isin(ids_to_keep)]
	print (df_f.shape)

	columns_to_keep = ['ConversationId', 'MessageText', 'queryPattern', 'judged_domain', 'JudgedTaskunderstandingIntent', 'JudgedTaskunderstandingMsgannotation', 
	'CuDomain', 'CuIntents', 'CuConstraints', 'CuConstraintsInWfSchemaAndFormat', 'AllCorrect']

	df_f = df_f[columns_to_keep]

	df_f.to_csv(file.rstrip('.tsv')+'_TS.tsv', sep='\t', index=False, encoding='utf-8')

	f1.writelines('{}\t{}\t{}\t{}\n'.format(item[0], item[1], F, T*1.0/item[1]))
	pat_1 = [item[0] for item in sorted_annotation if item[1]==1]
	df_t = df[df['queryPattern'].isin(pat_1)]
	counts = df_t.AllCorrect.value_counts()
	f1.writelines('{}\t{}\t{}\t{}\n'.format('PatternWithFreq1', len(pat_1), counts['F'] if 'F' in counts else 0, counts['T']*1.0/len(pat_1)))
	exit()

def checkSlotEqual(aa_slots, disa_slots):
	if len(aa_slots) !=len(disa_slots):
		return 0
	for key in aa_slots:
		if key not in disa_slots:
			return 0
	for key1 in disa_slots:
		if key1 not in aa_slots:
			return 0
	return 1
def checkSlotEqualStrict(aa_slots, disa_slots):
	if len(aa_slots) !=len(disa_slots):
		return 0
	for key in aa_slots:
		if key not in disa_slots:
			return 0
		elif aa_slots[key]!=disa_slots[key]:
			return 0
	for key1 in disa_slots:
		if key1 not in aa_slots:
			return 0
		elif disa_slots[key1] !=aa_slots[key1]:
			return 0
	return 1

def processFile(file):
	df = pd.read_csv(file, sep='\t', header=0, encoding = "utf-8")
	fussEqual = []

	for i, aa in enumerate(df['Slots']):
		aa_slots = processSlot(aa)
		disa_slots = processSlot(df['JSlots'][i])
		if df['Domain'][i] != df['JDomain'][i] or df['Intent'][i] != df['JIntent'][i]:
			fussEqual.append(0)
		else:
			if df['SimilarityScore'][i]>0.98:
				fussEqual.append(checkSlotEqualStrict(aa_slots, disa_slots))
			else:
				fussEqual.append(checkSlotEqual(aa_slots, disa_slots))

	df['FussEqual'] = fussEqual

	df.to_csv(file.rstrip('.tsv')+'_Post.tsv', sep='\t', index=False, encoding='utf-8')


if __name__=='__main__':
	file = 'AllDetails.2020_02_27_14_34_48_mv4_target_set_sample.tsv'

	#processFile(file)
	processAllDetail(file)




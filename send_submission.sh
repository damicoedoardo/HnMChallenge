echo Input submission name:
read submission_name
echo Input message:
read message
kaggle competitions submit -c h-and-m-personalized-fashion-recommendations -f submission/$submission_name.csv -m "$message"
python hnmchallenge/utils/good_luck.py
                                                                            
                                                                                                  
                                                                                                   

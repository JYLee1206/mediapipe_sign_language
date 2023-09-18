import cv2
import mediapipe as mp 
import joblib
import numpy as np 
import time
import tkinter as tk
from PIL import ImageTk, Image
__all__ = ["split_syllable_char", "split_syllables",
           "join_jamos", "join_jamos_char",
           "CHAR_INITIALS", "CHAR_MEDIALS", "CHAR_FINALS"]

import itertools

INITIAL = 0x001
MEDIAL = 0x010
FINAL = 0x100
CHAR_LISTS = {
    INITIAL: list(map(chr, [
        0x3131, 0x3132, 0x3134, 0x3137, 0x3138, 0x3139,
        0x3141, 0x3142, 0x3143, 0x3145, 0x3146, 0x3147,
        0x3148, 0x3149, 0x314a, 0x314b, 0x314c, 0x314d,
        0x314e
    ])),
    MEDIAL: list(map(chr, [
        0x314f, 0x3150, 0x3151, 0x3152, 0x3153, 0x3154,
        0x3155, 0x3156, 0x3157, 0x3158, 0x3159, 0x315a,
        0x315b, 0x315c, 0x315d, 0x315e, 0x315f, 0x3160,
        0x3161, 0x3162, 0x3163
    ])),
    FINAL: list(map(chr, [
        0x3131, 0x3132, 0x3133, 0x3134, 0x3135, 0x3136,
        0x3137, 0x3139, 0x313a, 0x313b, 0x313c, 0x313d,
        0x313e, 0x313f, 0x3140, 0x3141, 0x3142, 0x3144,
        0x3145, 0x3146, 0x3147, 0x3148, 0x314a, 0x314b,
        0x314c, 0x314d, 0x314e
    ]))
}
CHAR_INITIALS = CHAR_LISTS[INITIAL]
CHAR_MEDIALS = CHAR_LISTS[MEDIAL]
CHAR_FINALS = CHAR_LISTS[FINAL]
CHAR_SETS = {k: set(v) for k, v in CHAR_LISTS.items()}
CHARSET = set(itertools.chain(*CHAR_SETS.values()))
CHAR_INDICES = {k: {c: i for i, c in enumerate(v)}
                for k, v in CHAR_LISTS.items()}


def is_hangul_syllable(c):
    return 0xac00 <= ord(c) <= 0xd7a3  # Hangul Syllables
def is_hangul_jamo(c):
    return 0x1100 <= ord(c) <= 0x11ff  # Hangul Jamo
def is_hangul_compat_jamo(c):
    return 0x3130 <= ord(c) <= 0x318f  # Hangul Compatibility Jamo
def is_hangul_jamo_exta(c):
    return 0xa960 <= ord(c) <= 0xa97f  # Hangul Jamo Extended-A
def is_hangul_jamo_extb(c):
    return 0xd7b0 <= ord(c) <= 0xd7ff  # Hangul Jamo Extended-B
def is_hangul(c):
    return (is_hangul_syllable(c) or
            is_hangul_jamo(c) or
            is_hangul_compat_jamo(c) or
            is_hangul_jamo_exta(c) or
            is_hangul_jamo_extb(c))
def is_supported_hangul(c):
    return is_hangul_syllable(c) or is_hangul_compat_jamo(c)
def check_hangul(c, jamo_only=False):
    if not ((jamo_only or is_hangul_compat_jamo(c)) or is_supported_hangul(c)):
        raise ValueError(f"'{c}' is not a supported hangul character. "
                         f"'Hangul Syllables' (0xac00 ~ 0xd7a3) and "
                         f"'Hangul Compatibility Jamos' (0x3130 ~ 0x318f) are "
                         f"supported at the moment.")
def get_jamo_type(c):
    check_hangul(c)
    assert is_hangul_compat_jamo(c), f"not a jamo: {ord(c):x}"
    return sum(t for t, s in CHAR_SETS.items() if c in s)
def join_jamos_char(init, med, final=None):
    chars = (init, med, final)
    for c in filter(None, chars):
        check_hangul(c, jamo_only=True)

    idx = tuple(CHAR_INDICES[pos][c] if c is not None else c
                for pos, c in zip((INITIAL, MEDIAL, FINAL), chars))
    init_idx, med_idx, final_idx = idx
    # final index must be shifted once as
    # final index with 0 points to syllables without final
    final_idx = 0 if final_idx is None else final_idx + 1
    return chr(0xac00 + 28 * 21 * init_idx + 28 * med_idx + final_idx)
def join_jamos(s, ignore_err=True):
    last_t = 0
    queue = []
    new_string = ""

    def flush(n=0):
        new_queue = []
        while len(queue) > n:
            new_queue.append(queue.pop())
        if len(new_queue) == 1:
            if not ignore_err:
                raise ValueError(f"invalid jamo character: {new_queue[0]}")
            result = new_queue[0]
        elif len(new_queue) >= 2:
            try:
                result = join_jamos_char(*new_queue)
            except (ValueError, KeyError):
                # Invalid jamo combination
                if not ignore_err:
                    raise ValueError(f"invalid jamo characters: {new_queue}")
                result = "".join(new_queue)
        else:
            result = None
        return result

    for c in s:
        if c not in CHARSET:
            if queue:
                new_c = flush() + c
            else:
                new_c = c
            last_t = 0
        else:
            t = get_jamo_type(c)
            new_c = None
            if t & FINAL == FINAL:
                if not (last_t == MEDIAL):
                    new_c = flush()
            elif t == INITIAL:
                new_c = flush()
            elif t == MEDIAL:
                if last_t & INITIAL == INITIAL:
                    new_c = flush(1)
                else:
                    new_c = flush()
            last_t = t
            queue.insert(0, c)
        if new_c:
            new_string += new_c
    if queue:
        new_string += flush()
    return new_string

window = tk.Tk()
window.geometry('400x200')

# Create a label to display the image
image_label = tk.Label(window)
gesture_label = tk.Label(window, font=('Arial', 24))

def clear():
    jamo.clear()
    gesture_label.configure(text="")

button = tk.Button(window, text='초기화', command=clear)
button.place(x=100,y=30)
gesture_label.place(x=100, y=0)


def update_gui(image, gesture_text):
    
    image_pil = Image.fromarray(image)
    image_tk = ImageTk.PhotoImage(image_pil)

    
    image_label.configure(image=image_tk)
    
    jamo.append(gesture_text)
    gesture_label.configure(text=join_jamos("".join(jamo)))



gesture = {
	0:'ㄱ',1:'ㄴ',2:'ㄷ',3:'ㄹ',4:'ㅁ',5:'ㅂ',6:'ㅅ',7:'ㅇ',
	8:'ㅈ',9:'ㅊ',10:'ㅋ',11:'ㅌ',12:'ㅍ',13:'ㅎ',14:'ㅏ',
	15:'ㅑ',16:'ㅓ',17:'ㅕ',18:'ㅗ',19:'ㅛ',20:'ㅜ',21:'ㅠ',
	22:'ㅡ',23:'ㅣ',24:'ㅢ',25:'ㅚ',26:'ㅐ',27:'ㅒ', 28:'ㅔ', 29:'ㅖ', 30:'ㅘ', 31:'ㅙ'
	}
	
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
hands = mp_hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
startTime = time.time()
prev_index = 0
sentence = ''
recognizeDelay = 1
jamo = []
def data_clean(landmark):
  
  data = landmark[0]
  
  try:
    data = str(data)

    data = data.strip().split('\n')

    garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

    without_garbage = []

    for i in data:
        if i not in garbage:
            without_garbage.append(i)

    clean = []

    for i in without_garbage:
        i = i.strip()
        clean.append(i[2:])

    for i in range(0, len(clean)):
        clean[i] = float(clean[i])

    
    return([clean])

  except:
    return(np.zeros([1,63], dtype=int)[0])

while cap.isOpened():
  success, image = cap.read()
  
  image = cv2.flip(image, 1)
  
  if not success:
    break

  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  results = hands.process(image)

  # Draw the hand annotations on the image.
  image.flags.writeable = True

  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  
  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
          image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cleaned_landmark = data_clean(results.multi_hand_landmarks)
    #print(cleaned_landmark)

    if cleaned_landmark:
      clf = joblib.load('model.pkl')
      y_pred = clf.predict(cleaned_landmark)
      image = cv2.putText(image, str(y_pred[0]), (50,150), cv2.FONT_HERSHEY_SIMPLEX,  3, (0,0,255), 2, cv2.LINE_AA)
      index = int(y_pred)
      if index in gesture.keys():
          if index != prev_index:
            startTime = time.time()
            prev_index = index
          else:
              if time.time() - startTime > recognizeDelay:
                sentence += gesture[index]
                startTime = time.time()
                update_gui(image, gesture[index])
                
                
                
                
  window.update()              
  cv2.imshow('MediaPipe Hands', image)
  
  if cv2.waitKey(5) & 0xFF == 27:
    break
window.mainloop()
hands.close()
cap.release()
cv2.destroyAllWindows()


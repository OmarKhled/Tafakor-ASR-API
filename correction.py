import difflib
import editdistance

def closest_word(word: str, candidates: list[str]) -> str:
  closest = candidates[0]
  min_distance = editdistance.eval(word, closest)

  for candidate in candidates[1:]:
      distance = editdistance.eval(word, candidate)
      if distance < min_distance:
          min_distance = distance
          closest = candidate

  return closest

def spell_check(text: str, word_corpus: list[str]) -> str:
  corrected_words = [closest_word(word, word_corpus) for word in text.split()]
  return " ".join(corrected_words)


THRESOLD = 8
def correct_segmentation(segment: list[str], ground_truth: list[str], depth = 0):
  s = difflib.SequenceMatcher(None, segment, ground_truth)
  matching_blocks = s.get_matching_blocks()
  segments = []

  for index, current_segment in enumerate(matching_blocks):
    if index != len(matching_blocks) - 1:
      next_segment = matching_blocks[index + 1]

      matched_segment = [current_segment.b, current_segment.b + current_segment.size]
      segments.append(matched_segment)

      print(depth, current_segment, next_segment, matched_segment)

      if next_segment.size != 0:
        dropped_segment = [current_segment.a + current_segment.size, next_segment.a]
        dropped_segment_text = segment[dropped_segment[0]:dropped_segment[1]]

        dropped_segment_mappings = correct_segmentation(dropped_segment_text, ground_truth, depth + 1)

        print(depth, matched_segment, dropped_segment, dropped_segment_mappings)

        segments += dropped_segment_mappings
      else:
        last_segment = [current_segment.a + current_segment.size, len(segment)]
        last_segment_text = segment[last_segment[0]: last_segment[1]]

        dropped_segment_mappings = correct_segmentation(last_segment_text, ground_truth, depth + 1)

        print(depth, "last_segment", last_segment, dropped_segment_mappings)
        
        if len(dropped_segment_mappings) > 0 and (dropped_segment_mappings[0][0] <= segments[-1][1] and segments[-1][1] - dropped_segment_mappings[0][0] <= THRESOLD):
          segments += dropped_segment_mappings

  
  print("----segments post-process----")
  for index, current_segment in enumerate(segments):
    if index != len(segments) - 1:
      next_segment = segments[index + 1]
      if current_segment[1] < next_segment[0]:
        print(current_segment, next_segment)
        if next_segment[0] - current_segment[1] > THRESOLD:
          print(f"deleteing segments[{index + 1}] = {segments[index + 1]}")
          del segments[index + 1]
          current_segment = segments[index]
          next_segment = segments[index + 1]
        segments[index][1] = next_segment[0]
  print("-----------------------------")
  
  return segments

def refine_transcription(text: str, ground_truth: str):
  word_corpus = list(set(ground_truth.split(" ")))
  corrected_text = spell_check(text, word_corpus)

  segments = correct_segmentation(corrected_text.split(" "), ground_truth.split(" "))

  print(segments)
  final_text = ""
  for i in segments:
    final_text += " ".join(ground_truth.split(" ")[i[0]:i[1]]) 
    final_text += " "

  return final_text.strip()

  """
  final_segments = []

  if len(matching_blocks) == 2:
    return [matching_blocks[0].b, matching_blocks[0].b + matching_blocks[0].size]
  else:
    for index, s in enumerate(matching_blocks):
      if index == len(matching_blocks) - 1:
        return final_segments

      n_s = matching_blocks[index + 1]

      missing_range = [s.a + s.size, n_s.a]
      missing_text = [segment[i] for i in range(missing_range[0], missing_range[1])]

      # print(missing_text)

      if len(missing_text) == 0:
        if n_s.size > 0: final_segments.append([s.b, n_s.b])
      else:
        if n_s.size > 0:
          final_segments.append([s.b, n_s.b])
        else:
          final_segments.append([s.b, n_s.a])

        corrected_segment = correct_segmentation(missing_text, ground_truth)

        print(corrected_segment)

        if len(corrected_segment) > 0 and corrected_segment[0] <= n_s.b and corrected_segment[1] <= n_s.b + n_s.size or index == len(matching_blocks) - 2:
          final_segments.append(corrected_segment)
  """
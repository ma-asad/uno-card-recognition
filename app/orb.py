# orb.py

import cv2
import numpy as np
import glob
import os


class UNOCardDetector:
    def __init__(self, templates_path='app/static/img/*.jpg'):
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=7000)

        # FLANN parameters for ORB
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=14,
                            key_size=12,
                            multi_probe_level=2)
        search_params = dict(checks=80)

        # Initialize FLANN-based matcher
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Dictionary to hold cards and their templates
        self.cards = {}

        # Load all template images from the templates directory
        template_paths = glob.glob(templates_path)
        if not template_paths:
            print("No template images found in the templates directory.")
            exit()

        for template_path in template_paths:
            # Load the template image in grayscale
            template_img = cv2.imread(template_path, 0)
            if template_img is None:
                print(
                    f"Error: Template image {template_path} could not be loaded.")
                continue

            # Compute keypoints and descriptors for the template
            kp_template, des_template = self.orb.detectAndCompute(
                template_img, None)

            if des_template is None or len(des_template) < 2:
                print(
                    f"Warning: Not enough descriptors in template {template_path}.")
                continue

            # Extract the card name from the file path
            filename = os.path.basename(template_path)
            card_name = os.path.splitext(filename)[0]  # Remove file extension

            # Initialize the list for this card if not already done
            if card_name not in self.cards:
                self.cards[card_name] = []

            # Store the template information under the card name
            self.cards[card_name].append({
                'image': template_img,
                'keypoints': kp_template,
                'descriptors': des_template
            })

        print(f"Loaded templates for {len(self.cards)} cards.")

    def detect_card(self, frame):
        # Set thresholds
        min_area = 35000  # Adjust based on testing
        min_inliers = 15  # Minimum number of inliers required
        ratio_thresh = 0.75  # Lowe's ratio test threshold

        # Convert frame to grayscale for ORB
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the keypoints and descriptors in the current frame
        kp_frame, des_frame = self.orb.detectAndCompute(gray_frame, None)

        if des_frame is not None and len(des_frame) >= 2:
            # Keep track of the best match in this frame
            best_match_data = None

            for card_name, templates in self.cards.items():
                for template in templates:
                    des_template = template['descriptors']
                    kp_template = template['keypoints']
                    template_image = template['image']

                    # Ensure there are enough descriptors in the template
                    if des_template is None or len(des_template) < 2:
                        continue  # Skip this template if not enough descriptors

                    # Match descriptors between the template and the frame using KNN
                    matches = self.flann.knnMatch(des_template, des_frame, k=2)

                    # Apply Lowe's ratio test
                    good_matches = []
                    for match in matches:
                        if len(match) == 2:
                            m, n = match
                            if m.distance < ratio_thresh * n.distance:
                                good_matches.append(m)

                    # Proceed if enough good matches are found
                    if len(good_matches) > min_inliers:
                        # Extract locations of good matches
                        src_pts = np.float32(
                            [kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32(
                            [kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                        # Compute homography
                        M, mask = cv2.findHomography(
                            src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if M is not None:
                            matches_mask = mask.ravel().tolist()
                            h, w = template_image.shape
                            pts = np.float32(
                                [[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                            dst = cv2.perspectiveTransform(pts, M)
                            area = cv2.contourArea(dst)

                            # Calculate the number of inliers
                            num_inliers = np.sum(matches_mask)

                            # Update best match if criteria are met
                            if best_match_data is None or num_inliers > best_match_data['num_inliers']:
                                centroid = np.mean(dst, axis=0)
                                best_match_data = {
                                    'card_name': card_name,
                                    'dst': dst,
                                    'centroid': centroid,
                                    'area': area,
                                    'num_inliers': num_inliers
                                }
                        else:
                            pass  # Suppress output for better performance

            if best_match_data is not None:
                area = best_match_data['area']
                num_inliers = best_match_data['num_inliers']

                # Adjust the thresholds based on testing
                if area > min_area and num_inliers > min_inliers:
                    card_name = best_match_data['card_name']
                    print(
                        f"Detected card: {card_name}, Area: {area:.2f}, Inliers: {num_inliers}")
                    return card_name
                else:
                    print("Detection did not meet thresholds.")
                    return "Unknown Card"
            else:
                print("No matching cards found in the current frame.")
                return "Unknown Card"
        else:
            print("No descriptors found in the current frame.")
            return "Unknown Card"

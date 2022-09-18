import sys
import itertools
import numpy as np
import cv2 as cv

MAP_FILE = 'floorplan.png'

# Assign search area (SA) corner point locations based on image pixels.
SA1_CORNERS = (15, 15, 215, 180)  # (UL-X, UL-Y, LR-X, LR-Y)
SA2_CORNERS = (370, 190, 640, 388)
SA3_CORNERS = (15, 255, 180, 390)
SA4_CORNERS = (500, 16, 645, 122)


class Search:
    """Bayesian Search game with 4 search areas."""
    def __init__(self, name):
        self.name = name
        self.img = cv.imread(MAP_FILE)
        if self.img is None:
            print(f'Could not load map file {MAP_FILE}', file=sys.stderr)
            sys.exit(1)

    # Set placeholders for actual location
        self.area_actual = 0
        self.glasses_actual = [0, 0]  # in local area

        self.sa1 = self.img[SA1_CORNERS[1]: SA1_CORNERS[3],
                            SA1_CORNERS[0]: SA1_CORNERS[2]]

        self.sa2 = self.img[SA2_CORNERS[1]: SA2_CORNERS[3],
                            SA2_CORNERS[0]: SA2_CORNERS[2]]

        self.sa3 = self.img[SA3_CORNERS[1]: SA3_CORNERS[3],
                            SA3_CORNERS[0]: SA3_CORNERS[2]]

        self.sa4 = self.img[SA4_CORNERS[1]: SA4_CORNERS[3],
                            SA4_CORNERS[0]: SA4_CORNERS[2]]

        # Probabilities in every area
        self.p1 = 0.2
        self.p2 = 0.3
        self.p3 = 0.2
        self.p4 = 0.3

        # Search Effectiveness in every area
        self.se1 = 0
        self.se2 = 0
        self.se3 = 0
        self.se4 = 0

    def draw_map(self, last_known):
        """Display map, last know position and search areas."""

        # Search areas and numbers
        cv.rectangle(self.img, (SA1_CORNERS[0], SA1_CORNERS[1]),
                     (SA1_CORNERS[2], SA1_CORNERS[3]), (255, 255, 0), 2)
        cv.putText(self.img, '1',
                   (SA1_CORNERS[0] + 185, SA1_CORNERS[1] + 20),
                   cv.FONT_HERSHEY_PLAIN, 1, 0)

        cv.rectangle(self.img, (SA2_CORNERS[0], SA2_CORNERS[1]),
                     (SA2_CORNERS[2], SA2_CORNERS[3]), (255, 255, 0), 2)
        cv.putText(self.img, '2',
                   (SA2_CORNERS[0] + 3, SA2_CORNERS[1] + 15),
                   cv.FONT_HERSHEY_PLAIN, 1, 0)

        cv.rectangle(self.img, (SA3_CORNERS[0], SA3_CORNERS[1]),
                     (SA3_CORNERS[2], SA3_CORNERS[3]), (255, 255, 0), 2)
        cv.putText(self.img, '3',
                   (SA3_CORNERS[0] + 3, SA3_CORNERS[1] + 15),
                   cv.FONT_HERSHEY_PLAIN, 1, 0)

        cv.rectangle(self.img, (SA4_CORNERS[0], SA4_CORNERS[1]),
                     (SA4_CORNERS[2], SA4_CORNERS[3]), (255, 255, 0), 2)
        cv.putText(self.img, '4',
                   (SA4_CORNERS[0] + 3, SA4_CORNERS[1] + 15),
                   cv.FONT_HERSHEY_PLAIN, 1, 0)

        # last known location of the glasses.
        cv.putText(self.img, '+', last_known, cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255))
        cv.putText(self.img, '+ = Last known position', (400, 54),
                   cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv.putText(self.img, '* = Actual Position', (400, 74),
                   cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))

        cv.imshow('Search Area', self.img)
        cv.waitKey(500)

    def glasses_final_location(self, num_search_areas):
        """Return the actual x,y location of the missing glasses."""

        # Pick a search area at random.
        area = np.random.randint(1, num_search_areas + 1)

        # Convert local search area coordinates to map coordinates.
        x, y = None, None
        if area == 1:
            x = np.random.choice(SA1_CORNERS[2] - SA1_CORNERS[0]) + SA1_CORNERS[0]
            y = np.random.choice(SA1_CORNERS[3] - SA1_CORNERS[1]) + SA1_CORNERS[1]
            self.area_actual = 1
        elif area == 2:
            x = np.random.choice(SA2_CORNERS[2] - SA2_CORNERS[0]) + SA2_CORNERS[0]
            y = np.random.choice(SA2_CORNERS[3] - SA2_CORNERS[1]) + SA2_CORNERS[1]
            self.area_actual = 2
        elif area == 3:
            x = np.random.choice(SA3_CORNERS[2] - SA3_CORNERS[0]) + SA3_CORNERS[0]
            y = np.random.choice(SA3_CORNERS[3] - SA3_CORNERS[1]) + SA3_CORNERS[1]
            self.area_actual = 3
        elif area == 4:
            x = np.random.choice(SA4_CORNERS[2] - SA4_CORNERS[0]) + SA4_CORNERS[0]
            y = np.random.choice(SA4_CORNERS[3] - SA4_CORNERS[1]) + SA4_CORNERS[1]
            self.area_actual = 4
        return x, y

    def calc_search_effectiveness(self):
        """Set search effectiveness value per search area."""
        self.se1 = np.random.uniform(0.2, 0.9)
        self.se2 = np.random.uniform(0.2, 0.9)
        self.se3 = np.random.uniform(0.2, 0.9)
        self.se4 = np.random.uniform(0.2, 0.9)

    def conduct_search(self, area_num, area_array, effectiveness_prob):
        """Return search results and list of searched coordinates."""
        local_y_range = range(area_array.shape[0])
        local_x_range = range(area_array.shape[1])
        coords = list(itertools.product(local_x_range, local_y_range))
        np.random.shuffle(coords)
        coords = coords[:int(len(coords) * effectiveness_prob)]
        loc_actual = (self.glasses_actual[0], self.glasses_actual[1])
        if area_num == self.area_actual and loc_actual in coords:
            return f'Found in Area {area_num}', coords
        return 'Not Found', coords

    def revise_target_probs(self):
        """Update area target probabilities based on search effectiveness."""
        denominator = self.p1 * (1 - self.se1) + self.p2 * (1 - self.se2) + \
            self.p3 * (1 - self.se3) + self.p4 * (1 - self.se4)

        self.p1 = self.p1 * (1 - self.se1) / denominator
        self.p2 = self.p2 * (1 - self.se2) / denominator
        self.p3 = self.p3 * (1 - self.se3) / denominator
        self.p4 = self.p4 * (1 - self.se4) / denominator


def draw_menu(search_num):
    """Print menu of choices for conducting area searches."""
    print(f'\nSearch {search_num}')
    print(
        """
        Choose next areas to search:
        
        0 - Quit
        1 - Search Areas 1 & 2
        2 - Search Areas 1 & 3
        3 - Search Areas 1 & 4
        4 - Search Areas 2 & 3
        5 - Search Areas 2 & 4
        6 - Search Areas 3 & 4
        7 - Start Over
        
        """
    )


def main():
    app = Search('Where are my glasses?')
    app.draw_map(last_known=(160, 290))
    glasses_x, glasses_y = app.glasses_final_location(num_search_areas=4)
    print("-" * 65)
    print('\nInitial Target (P) Probabilities:')
    print(f"P1 = {app.p1}, P2 = {app.p2}, P3 = {app.p3}, P4 = {app.p4}")
    search_num = 1

    results_1, coords_1 = None, None
    results_2, coords_2 = None, None

    while True:
        app.calc_search_effectiveness()
        draw_menu(search_num)
        choice = int(input('Choice: '))

        if choice == 0:
            sys.exit()
        elif choice == 1:
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.se1)
            results_2, coords_2 = app.conduct_search(2, app.sa2, app.se2)
            app.se3 = 0
            app.se4 = 0

        elif choice == 2:
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.se1)
            results_2, coords_2 = app.conduct_search(3, app.sa3, app.se3)
            app.se2 = 0
            app.se4 = 0

        elif choice == 3:
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.se1)
            results_2, coords_2 = app.conduct_search(4, app.sa4, app.se4)
            app.se2 = 0
            app.se3 = 0

        elif choice == 4:
            results_1, coords_1 = app.conduct_search(2, app.sa2, app.se2)
            results_2, coords_2 = app.conduct_search(3, app.sa3, app.se3)
            app.se1 = 0
            app.se4 = 0

        elif choice == 5:
            results_1, coords_1 = app.conduct_search(2, app.sa2, app.se2)
            results_2, coords_2 = app.conduct_search(4, app.sa4, app.se4)
            app.se1 = 0
            app.se3 = 0

        elif choice == 6:
            results_1, coords_1 = app.conduct_search(3, app.sa3, app.se3)
            results_2, coords_2 = app.conduct_search(4, app.sa4, app.se4)
            app.se1 = 0
            app.se2 = 0

        elif choice == 7:
            main()

        else:
            print("\nSorry, but that is not a valid choice", file=sys.stderr)
            continue

        app.revise_target_probs()  # Use Bayes' rule to update target probs.

        print(f'Search {search_num} Results 1 = {results_1}')
        print(f'Search {search_num} Results 2 = {results_2}')
        print(f'Search {search_num} Effectiveness (E):')
        print(f'E1 = {app.se1}, E2 = {app.se2}, E3 = {app.se3}, E4 = {app.se4}')

        # Print target probabilities if glasses are not found else show position.
        if results_1 == 'Not Found' and results_2 == 'Not Found':
            print(f"New Target Probabilities (P) for search {search_num + 1}")
            print(f'P1 = {app.p1}, P2 = {app.p2}, P3 = {app.p3}, P4 = {app.p4}')
        else:
            cv.circle(app.img, (glasses_x, glasses_y), 3, (255, 0, 0), -1)
            cv.imshow('Search Area', app.img)
            cv.waitKey(1500)
            main()
        search_num += 1


if __name__ == '__main__':
    main()

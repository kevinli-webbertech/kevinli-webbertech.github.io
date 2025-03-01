# Midterm Project - Forensic Analysis with `The Sleuth Kit `

* The Sleuth Kit (TSK) is an open-source forensic toolkit that includes a collection of command-line tools for analyzing disk images and file systems.

## Summary of Findings
✅ Identified that `secret.pdf` was deleted.
✅ Successfully recovered the deleted file.
✅ Extracted timestamps showing when it was modified and deleted.
✅ Created a forensic timeline for investigation.

Please following the following steps to perform the analysis, and provide your screenshots and steps,

## Task 1: Download a sample forensic disk image from Digital Corpora.  (20 pts)

Download the `.tar.gz` tarball, and unzip it.

## Task 2: Run the mmls command to check the disk image structure. (10 pts)

## Task 3: List Files in NTFS or whatever file system format. (10 pts)
Let us delete one file. And please pay attention to its inode.

## Task 4: Recover Deleted File (20 pts)
Please prove that the file has come back.

## Task 5: Extract Metadata. (20 pts)

Extract metadata from the disk image and try to explain when the file was created and when it was deleted.

## Task 6: Generate a Timeline (20 pts)

   * Extract file system metadata and generate a txt file. (10 pts)
   * Generate a timeline CSV file from the above txt file. (10 pts)

In your final report, you should attach word document of showing your work of all these steps, and the CSV file for the timeline.